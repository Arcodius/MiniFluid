#include "cuda_renderer.h"

#include "camera.h"
#include "cuda/cuda_fluid_solver.h"
#include "profile/profiler.h"
#include "solids/sdf.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

// Checkpoint C debug view: true replaces the image with the motion vectors computed in the resolve.
constexpr bool kShowMotionVectors = false;

struct RenderParams {
    float3 bounds;
    float cellSize;
    float3 lightDirection;  // Unit vector pointing toward the sun
    float3 sunRadiance;
    float3 skyRadiance;
    float extinction;
    float albedo;
    float phaseG;
    float emissionStrength;
    float exposure;
    int lit;
    int emission;
    int floor;
};

constexpr float kPi = 3.14159265358979f;

__host__ __device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 multiply(float3 value, float scale) {
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__device__ float3 mix(float3 a, float3 b, float t) {
    return make_float3(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z));
}

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(float3 value) {
    const float inverseLength = rsqrtf(dot(value, value));
    return multiply(value, inverseLength);
}

__device__ float saturate(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ __forceinline__
float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ __forceinline__
float3 lerp(float3 a, float3 b, float t) {
    return make_float3(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t));
}

// t: how much the new value b contributes to the result.
__device__ __forceinline__
float4 lerp(float4 a, float4 b, float t) {
    return make_float4(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t), lerp(a.w, b.w, t));
}

__device__ float4 sampleBilinear(const float4* frame, float2 coordinate, const int width, const int height) {
    const int x0 = static_cast<int>(coordinate.x);
    const int y0 = static_cast<int>(coordinate.y);
    const int x1 = x0 == width - 1 ? x0 : x0 + 1;
    const int y1 = y0 == height - 1 ? y0 : y0 + 1;

    return lerp(
        lerp(frame[y0 * width + x0], frame[y1 * width + x0], coordinate.y - y0),
        lerp(frame[y0 * width + x1], frame[y1 * width + x1], coordinate.y - y0),
        coordinate.x - x0
    );
}

__device__ bool intersectVolume(
    float3 origin,
    float3 direction,
    float3 bounds,
    float& entry,
    float& exit) {
    entry = 0.0f;
    exit = FLT_MAX;
    const float origins[3] = {origin.x, origin.y, origin.z};
    const float directions[3] = {direction.x, direction.y, direction.z};
    const float extents[3] = {bounds.x, bounds.y, bounds.z};

    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(directions[axis]) < 1e-8f) {
            if (origins[axis] < 0.0f || origins[axis] > extents[axis]) {
                return false;
            }
            continue;
        }
        const float inverseDirection = 1.0f / directions[axis];
        float nearDistance = -origins[axis] * inverseDirection;
        float farDistance = (extents[axis] - origins[axis]) * inverseDirection;
        if (nearDistance > farDistance) {
            const float temp = nearDistance;
            nearDistance = farDistance;
            farDistance = temp;
        }
        entry = fmaxf(entry, nearDistance);
        exit = fminf(exit, farDistance);
        if (entry > exit) {
            return false;
        }
    }
    return exit > entry;
}

// Texel centers of the volume textures sit at (i + 0.5) * cellSize, so world coordinates
// map to unnormalized texture coordinates by a single division.
__device__ float sampleDensity(cudaTextureObject_t density, float3 p, float invCellSize) {
    return fmaxf(tex3D<float>(density, p.x * invCellSize, p.y * invCellSize, p.z * invCellSize), 0.0f);
}

__device__ float henyeyGreenstein(float cosTheta, float g) {
    const float g2 = g * g;
    const float denominator = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / (4.0f * kPi * denominator * sqrtf(denominator));
}

// Interleaved gradient noise, used to jitter ray origins and dither the output.
__device__ float gradientNoise(float x, float y) {
    const float f = 0.06711056f * x + 0.00583715f * y;
    const float g = 52.9829189f * (f - floorf(f));
    return g - floorf(g);
}

// Optical depth from p toward the light through the density texture.
__device__ float opticalDepthToLight(
    cudaTextureObject_t density, float3 p, const RenderParams& params, float stepCells) {
    float entry;
    float exit;
    if (!intersectVolume(p, params.lightDirection, params.bounds, entry, exit)) {
        return 0.0f;
    }
    const float invCellSize = 1.0f / params.cellSize;
    const float stepSize = params.cellSize * stepCells;
    const int steps = min(static_cast<int>(ceilf((exit - entry) / stepSize)), 1024);
    float tau = 0.0f;
    float t = entry + 0.5f * stepSize;
    for (int s = 0; s < steps && tau < 20.0f; ++s, t += stepSize) {
        tau += sampleDensity(density, add(p, multiply(params.lightDirection, t)), invCellSize)
            * params.extinction * stepSize;
    }
    return tau;
}

__device__ float sampleSdf(cudaTextureObject_t sdf, float3 p, float invCellSize) {
    return tex3D<float>(sdf, p.x * invCellSize, p.y * invCellSize, p.z * invCellSize);
}

// Sphere tracing against the signed distance texture. The texture only exists inside the simulation box, so the ray is first clipped to it. Returns true and the distance along the ray when the surface is reached before maxDistance.
__device__ bool traceSolid(
    cudaTextureObject_t sdf,
    float3 origin,
    float3 direction,
    float maxDistance,
    const RenderParams& params,
    float& hitDistance) {
    float entry;
    float exit;
    if (!intersectVolume(origin, direction, params.bounds, entry, exit)) {
        return false;
    }
    exit = fminf(exit, maxDistance);

    const float invCellSize = 1.0f / params.cellSize;
    const float hitEpsilon = 0.05f * params.cellSize;
    float t = entry;
    for (int step = 0; step < 128 && t < exit; ++step) {
        const float d = sampleSdf(sdf, add(origin, multiply(direction, t)), invCellSize);
        if (d < hitEpsilon) {
            hitDistance = t;
            return true;
        }
        // The distance field says nothing closer than d exists, so a step of d cannot tunnel through the surface. 0.9 leaves a margin for interpolation error in the texture.
        t += 0.9f * d;
    }
    return false;
}

// How much light reaches origin past the solid along direction, in [0, 1].
// A binary hit test sampled at voxel centers turns the smooth shadow cylinder into a staircase
// of voxels. Instead, the ray keeps marching through the solid to find the signed minimum
// distance to the surface along it, and a ramp one cell wide maps that to visibility. The
// result is continuous across the silhouette, so the filtered light volume places the shadow
// edge between voxel centers instead of snapping it to them.
__device__ float solidVisibility(
    cudaTextureObject_t sdf, float3 origin, float3 direction, const RenderParams& params) {
    float entry;
    float exit;
    if (!intersectVolume(origin, direction, params.bounds, entry, exit)) {
        return 1.0f;
    }

    const float invCellSize = 1.0f / params.cellSize;
    const float halfWidth = 0.5f * params.cellSize;
    // Grazing rays creep along the surface; a minimum step keeps them moving.
    const float minStep = 0.1f * params.cellSize;
    float minDistance = FLT_MAX;
    float t = entry;
    for (int step = 0; step < 256 && t < exit; ++step) {
        const float d = sampleSdf(sdf, add(origin, multiply(direction, t)), invCellSize);
        minDistance = fminf(minDistance, d);
        if (minDistance <= -halfWidth) {
            return 0.0f;  // Deep enough inside that the ramp is already at zero
        }
        // Outside, d is a safe step; inside, |d| moves toward the deepest point.
        t += fmaxf(0.9f * fabsf(d), minStep);
    }
    return saturate(0.5f + minDistance * invCellSize);
}

// The surface normal is the normalized gradient of the distance field (central differences).
__device__ float3 sdfNormal(cudaTextureObject_t sdf, float3 p, const RenderParams& params) {
    const float invCellSize = 1.0f / params.cellSize;
    const float e = params.cellSize;
    const float3 gradient = make_float3(
        sampleSdf(sdf, make_float3(p.x + e, p.y, p.z), invCellSize)
            - sampleSdf(sdf, make_float3(p.x - e, p.y, p.z), invCellSize),
        sampleSdf(sdf, make_float3(p.x, p.y + e, p.z), invCellSize)
            - sampleSdf(sdf, make_float3(p.x, p.y - e, p.z), invCellSize),
        sampleSdf(sdf, make_float3(p.x, p.y, p.z + e), invCellSize)
            - sampleSdf(sdf, make_float3(p.x, p.y, p.z - e), invCellSize));
    return normalize(gradient);
}

// ---------------------------------------------------------------------------
// Light volume
// ---------------------------------------------------------------------------

// Smoke optical depth from each cell straight up to the top of the domain: a cheap sky-occlusion term. The solid's share is computed separately as a visibility in lightDepthKernel.
__global__ void skyDepthKernel(
    GridView3D<const float> density, float* skyDepth, float extinctionTimesCell) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= density.nx || k >= density.nz) {
        return;
    }

    float accumulated = 0.0f;
    for (int j = density.ny - 1; j >= 0; --j) {
        const float value = fmaxf(density(i, j, k), 0.0f) * extinctionTimesCell;
        const size_t index = (static_cast<size_t>(k) * density.ny + j) * density.nx + i;
        skyDepth[index] = accumulated + 0.5f * value;
        accumulated += value;
    }
}

__global__ void lightDepthKernel(
    cudaTextureObject_t density,
    cudaTextureObject_t sdf,
    const float* skyDepth,
    cudaSurfaceObject_t lightDepth,
    int nx,
    int ny,
    int nz,
    RenderParams params) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    const float3 p = make_float3(
        (i + 0.5f) * params.cellSize,
        (j + 0.5f) * params.cellSize,
        (k + 0.5f) * params.cellSize);
    // Smoke and solid are stored separately: the solid as a visibility in [0, 1], because linearly filtering a huge "opaque" optical depth would make exp(-tau) collapse to zero as soon as any neighboring voxel is blocked, which hardens the edge into voxel steps.
    const float sunVisibility = solidVisibility(sdf, p, params.lightDirection, params);
    const float skyVisibility = solidVisibility(sdf, p, make_float3(0.0f, 1.0f, 0.0f), params);
    // Computed even when fully shadowed, so filtering next to the shadow edge blends real depths.
    const float tau = opticalDepthToLight(density, p, params, 1.0f);
    const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
    surf3Dwrite(make_float4(tau, skyDepth[index], sunVisibility, skyVisibility),
        lightDepth, i * sizeof(float4), j, k);
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

__device__ float3 skyColor(float3 direction, const RenderParams& params) {
    const float3 horizon = make_float3(0.78f, 0.80f, 0.84f);
    const float3 zenith = make_float3(0.30f, 0.46f, 0.78f);
    const float elevation = saturate(direction.y);
    float3 color = mix(horizon, zenith, sqrtf(elevation));
    if (direction.y < 0.0f) {
        color = mix(horizon, make_float3(0.42f, 0.42f, 0.44f), saturate(-direction.y * 4.0f));
    }
    color = multiply(color, 0.9f);

    const float sunAmount = saturate(dot(direction, params.lightDirection));
    const float halo = powf(sunAmount, 12.0f) * 0.06f + powf(sunAmount, 900.0f) * 4.0f;
    return add(color, multiply(params.sunRadiance, halo));
}

__device__ float3 floorColor(
    float3 p,
    float3 direction,
    float distance,
    cudaTextureObject_t density,
    cudaTextureObject_t lightDepth,
    cudaTextureObject_t sdf,
    const RenderParams& params) {
    const float tile = 0.25f;
    const int checker = (static_cast<int>(floorf(p.x / tile)) + static_cast<int>(floorf(p.z / tile))) & 1;
    const float contrast = expf(-distance * 0.35f);
    const float albedo = 0.34f + (checker ? 0.05f : -0.02f) * contrast;

    // The same soft visibility as the light volume, traced per pixel, so the floor shadow and
    // the shadow in the smoke have matching edges.
    const float shadow = expf(-opticalDepthToLight(density, p, params, 1.5f))
        * solidVisibility(sdf, p, params.lightDirection, params);
    float occlusion = 1.0f;
    if (p.x > 0.0f && p.z > 0.0f && p.x < params.bounds.x && p.z < params.bounds.z) {
        const float invCellSize = 1.0f / params.cellSize;
        const float4 depth = tex3D<float4>(lightDepth,
            p.x * invCellSize, 0.5f, p.z * invCellSize);
        occlusion = expf(-0.35f * depth.y) * depth.w;
    }

    const float sunFactor = fmaxf(params.lightDirection.y, 0.0f) * shadow;
    const float3 lit = add(
        multiply(params.sunRadiance, sunFactor),
        multiply(params.skyRadiance, occlusion));
    const float3 color = multiply(lit, albedo);

    const float haze = 1.0f - expf(-distance * 0.12f);
    return mix(color, skyColor(direction, params), haze);
}

// Shading of the solid at hit point p: Lambert sun light (dimmed by smoke between the surface
// and the sun) plus a hemispherical sky term, then the same distance haze as the floor.
__device__ float3 solidColor(
    float3 p,
    float3 direction,
    float distance,
    cudaTextureObject_t density,
    cudaTextureObject_t sdf,
    const RenderParams& params) {
    const float3 albedo = make_float3(0.62f, 0.34f, 0.26f);
    const float3 normal = sdfNormal(sdf, p, params);

    const float smokeShadow = expf(-opticalDepthToLight(density, p, params, 1.5f));
    const float diffuse = fmaxf(dot(normal, params.lightDirection), 0.0f) * smokeShadow;
    const float skyAmount = 0.5f + 0.5f * normal.y;  // Facing up sees more sky than facing down

    const float3 lit = add(
        multiply(params.sunRadiance, diffuse),
        multiply(params.skyRadiance, skyAmount));
    const float3 color = make_float3(lit.x * albedo.x, lit.y * albedo.y, lit.z * albedo.z);

    const float haze = 1.0f - expf(-distance * 0.12f);
    return mix(color, skyColor(direction, params), haze);
}

__device__ float3 emissionColor(float temperature) {
    const float t = saturate((temperature - 0.15f) / 0.85f);
    const float t2 = t * t;
    return make_float3(t2 * 1.0f, t2 * t * 0.45f, t2 * t2 * 0.12f);
}

__device__ float3 toneMap(float3 color, float exposure) {
    const float channels[3] = {color.x * exposure, color.y * exposure, color.z * exposure};
    float mapped[3];
    for (int c = 0; c < 3; ++c) {
        const float x = fmaxf(channels[c], 0.0f);
        const float aces = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
        mapped[c] = powf(saturate(aces), 1.0f / 2.2f);
    }
    return make_float3(mapped[0], mapped[1], mapped[2]);
}

__device__ void generateRay(const int x, const int y, const int width, const int height, RenderCamera camera, float3& origin, float3& direction) {
    const float ndcX = 2.0f * (static_cast<float>(x) + 0.5f) / width - 1.0f;
    const float ndcY = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / height;
    origin = camera.position;
    if (camera.perspective) {
        const float aspect = static_cast<float>(width) / height;
        direction = normalize(add(
            add(camera.front,
                multiply(camera.right, ndcX * aspect * camera.tangentHalfFov)),
            multiply(camera.up, ndcY * camera.tangentHalfFov)));
    } else {
        origin = add(origin, add(
            multiply(camera.right, ndcX * camera.halfWidth),
            multiply(camera.up, ndcY * camera.halfHeight)));
        direction = camera.front;
    }
}

// The inverse of generateRay: where the point seen along (origin, direction) at 1 / invDepth was in the image of a perspective camera, in the same pixel coordinates as generateRay's x and y: the center of pixel (x, y) comes back as exactly (x, y). Returns false when the point is behind it.
// The point is P = origin + direction / invDepth. A perspective projection does not change when the vector it projects is scaled, so P - camera.position is multiplied by invDepth first. That keeps the sky (invDepth = 0, a point at infinity) finite: it projects its direction only.
__device__ bool projectToCamera(
    float3 origin, float3 direction, float invDepth, RenderCamera camera,
    int width, int height, float2& pixel) {
    if (!camera.perspective) {
        return false;  // Only perspective cameras are reprojected; resolveKernel handles a still orthographic one
    }
    const float3 v = add(multiply(subtract(origin, camera.position), invDepth), direction);
    const float z = dot(v, camera.front);
    if (z <= 1e-6f) {
        return false;
    }
    const float aspect = static_cast<float>(width) / height;
    const float ndcX = dot(v, camera.right) / (z * camera.tangentHalfFov * aspect);
    const float ndcY = dot(v, camera.up) / (z * camera.tangentHalfFov);
    // generateRay solved for x and y
    pixel.x = (ndcX + 1.0f) * 0.5f * width - 0.5f;
    pixel.y = (1.0f - ndcY) * 0.5f * height - 0.5f;
    return true;
}

__global__ void renderKernel(
    cudaTextureObject_t density,
    cudaTextureObject_t temperature,
    cudaTextureObject_t lightDepth,
    cudaTextureObject_t sdf,
    RenderParams params,
    RenderCamera camera,
    int width, int height, int frameIndex,
    float4* hdrPixels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 direction;
    float3 origin;
    generateRay(x, y, width, height, camera, origin, direction);

    const float shift = 5.588238f * float(frameIndex % 64);
    const float jitter = gradientNoise(static_cast<float>(x) + shift, static_cast<float>(y) + shift);
    const float invCellSize = 1.0f / params.cellSize;
    float floorDistance = FLT_MAX;
    if (params.lit && params.floor && direction.y < -1e-6f && origin.y > 0.0f) {
        floorDistance = -origin.y / direction.y;
    }

    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    float entry;
    float exit;
    // Nearest opaque surface along the ray. The solid is only searched for up to the floor, so
    // whichever of the two is hit first wins.
    float solidDistance = FLT_MAX;
    if (params.lit) {
        float hit;
        if (traceSolid(sdf, origin, direction, floorDistance, params, hit)) {
            solidDistance = hit;
        }
    }
    float invDepth = 0.0f;
    if (intersectVolume(origin, direction, params.bounds, entry, exit)) {
        // Smoke is only integrated in front of the opaque surface, which is what hides it behind.
        const float surfaceDistance = fminf(floorDistance, solidDistance);
        exit = fminf(exit, surfaceDistance);
        const float stepSize = params.cellSize * 10.0f;
        const int maxSteps = static_cast<int>(ceilf((exit - entry) / stepSize));
        float distance = fmaxf(entry + stepSize * jitter, 1e-6f); // to avoid zero distance in division

        const float cosTheta = dot(direction, params.lightDirection);
        // Multiple scattering approximation: a sum of octaves with progressively weaker
        // extinction and less anisotropic phase (Wrenninge et al.).
        // Phases are scaled by 4*pi so that an isotropic medium has a phase of one.
        float phases[3];
        for (int octave = 0; octave < 3; ++octave) {
            const float lobeG = params.phaseG * powf(0.5f, static_cast<float>(octave));
            phases[octave] = 4.0f * kPi * (0.8f * henyeyGreenstein(cosTheta, lobeG)
                + 0.2f * henyeyGreenstein(cosTheta, -0.3f * lobeG));
        }

        for (int sample = 0; sample < maxSteps && transmittance > 0.003f;
             ++sample, distance += stepSize) {
            const float3 p = add(origin, multiply(direction, distance));
            const float value = sampleDensity(density, p, invCellSize);
            float3 emitted = make_float3(0.0f, 0.0f, 0.0f);
            if (params.emission) {
                const float heat = tex3D<float>(temperature,
                    p.x * invCellSize, p.y * invCellSize, p.z * invCellSize);
                emitted = multiply(emissionColor(heat), params.emissionStrength * stepSize);
            }
            if (value < 1e-4f) {
                radiance = add(radiance, multiply(emitted, transmittance));
                continue;
            }

            if (!params.lit) {
                const float alpha = 1.0f - expf(-value * params.extinction * stepSize);
                const float contribution = transmittance * alpha;
                invDepth += contribution / distance;
                radiance = add(radiance, multiply(make_float3(0.92f, 0.96f, 1.0f), contribution));
                transmittance *= 1.0f - alpha;
                continue;
            }

            const float4 depth = tex3D<float4>(lightDepth,
                p.x * invCellSize, p.y * invCellSize, p.z * invCellSize);
            float sun = 0.0f;
            float weight = 1.0f;
            float extinctionScale = 1.0f;
            for (int octave = 0; octave < 3; ++octave) {
                sun += weight * phases[octave] * expf(-depth.x * extinctionScale);
                weight *= 0.3f;
                extinctionScale *= 0.5f;
            }
            sun *= depth.z;  // The solid blocks every octave alike
            const float skyVisibility = expf(-depth.y) * depth.w;
            const float3 inScattered = multiply(add(
                multiply(params.sunRadiance, sun),
                multiply(params.skyRadiance, skyVisibility)),
                params.albedo);

            // Analytic integration of constant in-scattering over the step.
            const float stepTransmittance = expf(-value * params.extinction * stepSize);
            radiance = add(radiance, multiply(
                add(multiply(inScattered, 1.0f - stepTransmittance), emitted), transmittance));
            invDepth += transmittance * (1.0f - stepTransmittance) / distance;
            transmittance *= stepTransmittance;
        }
    }

    if (params.lit && transmittance > 0.003f) {
        float3 background;
        if (solidDistance < FLT_MAX) {
            const float3 p = add(origin, multiply(direction, solidDistance));
            background = solidColor(p, direction, solidDistance, density, sdf, params);
            invDepth += transmittance / solidDistance;
        } else if (floorDistance < FLT_MAX) {
            const float3 p = add(origin, multiply(direction, floorDistance));
            background = floorColor(p, direction, floorDistance, density, lightDepth, sdf, params);
            invDepth += transmittance / floorDistance;
        } else {
            background = skyColor(direction, params);
        }
        radiance = add(radiance, multiply(background, transmittance));
    }

    hdrPixels[y * width + x] = make_float4(radiance.x, radiance.y, radiance.z, invDepth);
}

// previousCameraValid: false on the first frame, when there is no previous camera to project into.
// cameraUnchanged: the previous and the current camera are identical. Orthographic cameras are not
// reprojected, so for them this is the only case in which the history still lines up.
__global__ void resolveKernel(const float4* current, const float4* previousHistory, float4* currentHistory, uint8_t* pixels, RenderParams params, RenderCamera previousCamera, bool previousCameraValid, bool cameraUnchanged, RenderCamera currentCamera, int width, int height, bool historyReset, const int maxFrames) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int index = y * width + x;
    const float4 hdr = current[index];

    // Where this pixel's content was in the previous frame. hdr.w is the inverse depth, not the depth.
    float3 origin;
    float3 direction;
    generateRay(x, y, width, height, currentCamera, origin, direction);
    float2 previousPixel = make_float2(0.0f, 0.0f);
    bool reprojected = false;
    if (previousCameraValid) {
        if (previousCamera.perspective) {
            reprojected = projectToCamera(
                origin, direction, hdr.w, previousCamera, width, height, previousPixel);
        } else if (cameraUnchanged) {
            // A still orthographic camera: every pixel sees what it saw last frame.
            previousPixel = make_float2(static_cast<float>(x), static_cast<float>(y));
            reprojected = true;
        }
    }
    const bool onScreen = reprojected
        && previousPixel.x >= 0.0f && previousPixel.x <= width - 1.0f
        && previousPixel.y >= 0.0f && previousPixel.y <= height - 1.0f;
    const bool usable = !historyReset && onScreen;

    if (kShowMotionVectors) {
        // Checkpoint C: red = horizontal motion, green = vertical, scaled so 1/16 pixel is visible.
        // A still camera must be black everywhere. Blue: no usable history position.
        const float motionScale = 2.0f;
        const int byte = index * 4;
        const float red = onScreen ? saturate(fabsf(previousPixel.x - x) * motionScale) : 0.0f;
        const float green = onScreen ? saturate(fabsf(previousPixel.y - y) * motionScale) : 0.0f;
        pixels[byte + 0] = static_cast<uint8_t>(red * 255.0f + 0.5f);
        pixels[byte + 1] = static_cast<uint8_t>(green * 255.0f + 0.5f);
        pixels[byte + 2] = onScreen ? 0 : 255;
        pixels[byte + 3] = 255;
        return;
    }

    // On a reset the history is overwritten, not blended: it may be uninitialized memory, and lerp(NaN, hdr, 1) is still NaN, which would then stick in the history forever.
    const float4 h = usable ? sampleBilinear(previousHistory, previousPixel, width, height) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float n = usable ? h.w : 0;
    const float blend = 1.0f / (n + 1);
    const float3 rgb = usable ? lerp(make_float3(h.x, h.y, h.z), make_float3(hdr.x, hdr.y, hdr.z), blend) : make_float3(hdr.x, hdr.y, hdr.z);
    const float newN = fminf(n + 1, maxFrames);
    currentHistory[index] = make_float4(rgb.x, rgb.y, rgb.z, newN);
    
    const float3 color = params.lit ? toneMap(rgb, params.exposure) : rgb;
    const float dither = (gradientNoise(x + 17.0f, y + 59.0f) - 0.5f) / 255.0f;
    int byte = index * 4;
    pixels[byte + 0] = static_cast<uint8_t>(saturate(color.x + dither) * 255.0f + 0.5f);
    pixels[byte + 1] = static_cast<uint8_t>(saturate(color.y + dither) * 255.0f + 0.5f);
    pixels[byte + 2] = static_cast<uint8_t>(saturate(color.z + dither) * 255.0f + 0.5f);
    pixels[byte + 3] = 255;
}

float3 toFloat3(const Vector3& value) {
    return make_float3(value.x, value.y, value.z);
}

bool differ(float3 a, float3 b) {
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

bool sameCamera(const RenderCamera& a, const RenderCamera& b) {
    return !differ(a.position, b.position) && !differ(a.front, b.front)
        && !differ(a.right, b.right) && !differ(a.up, b.up)
        && a.tangentHalfFov == b.tangentHalfFov
        && a.halfWidth == b.halfWidth && a.halfHeight == b.halfHeight
        && a.perspective == b.perspective;
}

// Whether two settings would produce different HDR radiance. Exposure is left out on purpose:
// it is applied in the resolve, after the accumulation, so changing it needs no new history.
bool radianceSettingsDiffer(const LightingSettings& a, const LightingSettings& b) {
    return a.lightAzimuth != b.lightAzimuth || a.lightElevation != b.lightElevation
        || differ(a.sunColor, b.sunColor) || a.sunIntensity != b.sunIntensity
        || differ(a.skyColor, b.skyColor) || a.ambientIntensity != b.ambientIntensity
        || a.extinction != b.extinction || a.albedo != b.albedo || a.phaseG != b.phaseG
        || a.emissionStrength != b.emissionStrength
        || a.lit != b.lit || a.emission != b.emission || a.floor != b.floor;
}

}

VolumeTexture::VolumeTexture(
    int nx, int ny, int nz, cudaChannelFormatDesc format, bool writable, bool clampToEdge)
    : nx_(nx), ny_(ny), nz_(nz) {
    CUDA_CHECK(cudaMalloc3DArray(&array_, &format, make_cudaExtent(nx, ny, nz),
        writable ? cudaArraySurfaceLoadStore : cudaArrayDefault));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array_;

    cudaTextureDesc description{};
    const cudaTextureAddressMode addressMode =
        clampToEdge ? cudaAddressModeClamp : cudaAddressModeBorder;
    description.addressMode[0] = addressMode;
    description.addressMode[1] = addressMode;
    description.addressMode[2] = addressMode;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 0;
    try {
        CUDA_CHECK(cudaCreateTextureObject(&texture_, &resource, &description, nullptr));
        if (writable) {
            CUDA_CHECK(cudaCreateSurfaceObject(&surface_, &resource));
        }
    } catch (...) {
        releaseNoexcept();
        throw;
    }
}

VolumeTexture::~VolumeTexture() noexcept {
    releaseNoexcept();
}

VolumeTexture::VolumeTexture(VolumeTexture&& other) noexcept
    : array_(std::exchange(other.array_, nullptr)),
      texture_(std::exchange(other.texture_, 0)),
      surface_(std::exchange(other.surface_, 0)),
      nx_(other.nx_), ny_(other.ny_), nz_(other.nz_) {}

VolumeTexture& VolumeTexture::operator=(VolumeTexture&& other) noexcept {
    if (this != &other) {
        releaseNoexcept();
        array_ = std::exchange(other.array_, nullptr);
        texture_ = std::exchange(other.texture_, 0);
        surface_ = std::exchange(other.surface_, 0);
        nx_ = other.nx_;
        ny_ = other.ny_;
        nz_ = other.nz_;
    }
    return *this;
}

void VolumeTexture::releaseNoexcept() noexcept {
    if (surface_ != 0) {
        reportCudaCleanupError(cudaDestroySurfaceObject(surface_));
        surface_ = 0;
    }
    if (texture_ != 0) {
        reportCudaCleanupError(cudaDestroyTextureObject(texture_));
        texture_ = 0;
    }
    if (array_ != nullptr) {
        reportCudaCleanupError(cudaFreeArray(array_));
        array_ = nullptr;
    }
}

void VolumeTexture::copyFrom(GridView3D<const float> source) {
    if (source.nx != nx_ || source.ny != ny_ || source.nz != nz_) {
        throw std::invalid_argument("Volume texture and source grid dimensions do not match");
    }
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(source.data), source.nx * sizeof(float), source.nx, source.ny);
    copy.dstArray = array_;
    copy.extent = make_cudaExtent(source.nx, source.ny, source.nz);
    copy.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copy));
}

CudaRenderer::CudaRenderer(
    CudaFluidSolver& solver,
    Camera& camera,
    std::pair<size_t, size_t> resolution)
    : camera_(camera), solver_(solver),
      density_(solver.nx(), solver.ny(), solver.nz(), cudaCreateChannelDesc<float>(), false),
      temperature_(solver.nx(), solver.ny(), solver.nz(), cudaCreateChannelDesc<float>(), false),
      lightDepth_(solver.nx(), solver.ny(), solver.nz(), cudaCreateChannelDesc<float4>(), true),
      sdf_(solver.nx(), solver.ny(), solver.nz(), cudaCreateChannelDesc<float>(), false, true),
      skyDepth_(checkedElementCount(solver.nx(), solver.ny(), solver.nz())),
      frameIndex_(0) {
    setResolution(resolution.first, resolution.second);
}

void CudaRenderer::setResolution(size_t w, size_t h) {
    if (w == 0 || h == 0
        || w > static_cast<size_t>(std::numeric_limits<int>::max())
        || h > static_cast<size_t>(std::numeric_limits<int>::max())
        || w > std::numeric_limits<size_t>::max() / h
        || w * h > std::numeric_limits<size_t>::max() / 4) {
        throw std::invalid_argument("CUDA render resolution must be positive and fit in memory");
    }
    if (w == width_ && h == height_) {
        return;
    }

    DeviceBuffer<uint8_t> pixels(w * h * 4);
    DeviceBuffer<float4> hdrPixels(w * h);
    DeviceBuffer<float4> historyHdrPixels0(w * h);
    DeviceBuffer<float4> historyHdrPixels1(w * h);
    devicePixels_ = std::move(pixels);
    hdrPixels_ = std::move(hdrPixels);
    historyHdrPixels0_ = std::move(historyHdrPixels0);
    historyHdrPixels1_ = std::move(historyHdrPixels1);
    previousCameraValid_ = false;  // The previous frame had another size, so its pixels do not line up
    width_ = w;
    height_ = h;
}
std::pair<size_t, size_t> CudaRenderer::getResolution() const {
    return std::pair<size_t, size_t>(width_, height_);
}

namespace {

RenderParams makeRenderParams(const CudaFluidSolver& solver, const LightingSettings& lighting) {
    RenderParams params{};
    params.cellSize = solver.spacing();
    params.bounds = make_float3(
        solver.nx() * params.cellSize,
        solver.ny() * params.cellSize,
        solver.nz() * params.cellSize);
    const float azimuth = DirectX::XMConvertToRadians(lighting.lightAzimuth);
    const float elevation = DirectX::XMConvertToRadians(lighting.lightElevation);
    params.lightDirection = make_float3(
        std::cos(elevation) * std::cos(azimuth),
        std::sin(elevation),
        std::cos(elevation) * std::sin(azimuth));
    params.sunRadiance = multiply(lighting.sunColor, lighting.sunIntensity);
    params.skyRadiance = multiply(lighting.skyColor, lighting.ambientIntensity);
    params.extinction = lighting.extinction;
    params.albedo = lighting.albedo;
    params.phaseG = lighting.phaseG;
    params.emissionStrength = lighting.emissionStrength;
    params.exposure = lighting.exposure;
    params.lit = lighting.lit ? 1 : 0;
    params.emission = lighting.emission ? 1 : 0;
    params.floor = lighting.floor ? 1 : 0;
    return params;
}

}

void CudaRenderer::updateVolumes() {
    const uint64_t solverVersion = solver_.version();
    const GridView3D<const float> density = solver_.densityView();
    if (densityVersion_ != solverVersion) {
        density_.copyFrom(density);
        sdf_.copyFrom(solver_.sdfView());
        densityVersion_ = solverVersion;
    }
    // Skipped while emission is off, so the version stays old and turning emission back on uploads the current temperature instead of showing the one from when it was turned off.
    if (lighting_.emission && temperatureVersion_ != solverVersion) {
        temperature_.copyFrom(solver_.temperatureView());
        temperatureVersion_ = solverVersion;
    }
    // Unlit rendering never reads the light volume. lightVolumeInputs_ is left as it was, so it
    // still describes what the volume holds, and switching back to lit rebuilds it only if needed.
    if (!lighting_.lit) {
        return;
    }

    const LightVolumeInputs inputs{
        solverVersion, lighting_.lightAzimuth, lighting_.lightElevation, lighting_.extinction};
    if (inputs == lightVolumeInputs_) {
        return;
    }
    lightVolumeInputs_ = inputs;

    const RenderParams params = makeRenderParams(solver_, lighting_);
    const dim3 columnThreads(16, 16);
    const dim3 columnBlocks(
        (density.nx + columnThreads.x - 1) / columnThreads.x,
        (density.nz + columnThreads.y - 1) / columnThreads.y);
    skyDepthKernel<<<columnBlocks, columnThreads>>>(
        density, skyDepth_.data(), params.extinction * params.cellSize);
    CUDA_CHECK(cudaGetLastError());

    const dim3 threads(8, 8, 4);
    const dim3 blocks(
        (density.nx + threads.x - 1) / threads.x,
        (density.ny + threads.y - 1) / threads.y,
        (density.nz + threads.z - 1) / threads.z);
    lightDepthKernel<<<blocks, threads>>>(
        density_.texture(), sdf_.texture(), skyDepth_.data(), lightDepth_.surface(),
        density.nx, density.ny, density.nz, params);
    CUDA_CHECK(cudaGetLastError());
}

// Check and update parameters
bool CudaRenderer::historyNeedsReset() {
    // Pull-style change detection: each source keeps its own version, and the renderer
    // remembers the version its history was built from. Nobody has to notify the renderer.
    const uint64_t solverVersion = solver_.version();
    if (solverVersion != historySolverVersion_
        || radianceSettingsDiffer(lighting_, historyLighting_)) {
        historySolverVersion_ = solverVersion;
        historyLighting_ = lighting_;
        return true;
    }
    return false;
}

void CudaRenderer::render(uint8_t* destination, size_t destinationSize) {
    const size_t pixelBytes = width_ * height_ * 4;
    if (destination == nullptr || destinationSize != pixelBytes) {
        throw std::invalid_argument("CUDA render destination size does not match the resolution");
    }

    RenderCamera renderCamera{};
    renderCamera.position = toFloat3(camera_.getPosition());
    renderCamera.front = toFloat3(camera_.getFront());
    renderCamera.right = toFloat3(camera_.getRight());
    renderCamera.up = toFloat3(camera_.getUp());
    renderCamera.perspective = camera_.isPerspective() ? 1 : 0;
    if (renderCamera.perspective) {
        renderCamera.tangentHalfFov = std::tan(
            DirectX::XMConvertToRadians(camera_.getFovY()) * 0.5f);
    } else {
        renderCamera.halfHeight = camera_.getOrthographicZoom() * 0.5f;
        renderCamera.halfWidth = renderCamera.halfHeight
            * static_cast<float>(width_) / static_cast<float>(height_);
    }

    PROFILE_GPU(GpuStage::Volumes, updateVolumes());

    constexpr int tile = 16;
    const dim3 threads(tile, tile);
    const dim3 blocks(
        (static_cast<unsigned int>(width_) + tile - 1) / tile,
        (static_cast<unsigned int>(height_) + tile - 1) / tile);
    RenderParams params = makeRenderParams(solver_, lighting_);
    {
        const GpuScope scope(GpuStage::RayMarch);
        renderKernel<<<blocks, threads>>>(
            density_.texture(), temperature_.texture(), lightDepth_.texture(), sdf_.texture(),
            params, renderCamera,
            static_cast<int>(width_), static_cast<int>(height_), frameIndex_, hdrPixels_.data());
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const GpuScope scope(GpuStage::Resolve);
        resolveKernel<<<blocks, threads>>>(hdrPixels_.data(), historyHdrPixels0_.data(),  historyHdrPixels1_.data(),devicePixels_.data(), params,
            previousCamera_, previousCameraValid_, sameCamera(previousCamera_, renderCamera), renderCamera,
            static_cast<int>(width_), static_cast<int>(height_), historyNeedsReset(), kMaxAccumulatedFrames);
        CUDA_CHECK(cudaGetLastError());
    }
    std::swap(historyHdrPixels0_, historyHdrPixels1_);
    PROFILE_GPU(GpuStage::Readback, CUDA_CHECK(cudaMemcpy(
        destination, devicePixels_.data(), pixelBytes, cudaMemcpyDeviceToHost)));
    previousCamera_ = renderCamera;
    previousCameraValid_ = true;
    frameIndex_++;
}
