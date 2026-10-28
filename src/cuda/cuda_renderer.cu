#include "cuda_renderer.h"

#include "camera.h"
#include "cuda/cuda_fluid_solver.h"

#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

struct RenderCamera {
    float3 position;
    float3 front;
    float3 right;
    float3 up;
    float tangentHalfFov;
    float halfWidth;
    float halfHeight;
    int perspective;
};

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

// ---------------------------------------------------------------------------
// Light volume
// ---------------------------------------------------------------------------

// Optical depth from each cell straight up to the top of the domain: a cheap sky-occlusion term.
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
    const float tau = opticalDepthToLight(density, p, params, 1.0f);
    const size_t index = (static_cast<size_t>(k) * ny + j) * nx + i;
    surf3Dwrite(make_float2(tau, skyDepth[index]), lightDepth, i * sizeof(float2), j, k);
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
    const RenderParams& params) {
    const float tile = 0.25f;
    const int checker = (static_cast<int>(floorf(p.x / tile)) + static_cast<int>(floorf(p.z / tile))) & 1;
    const float contrast = expf(-distance * 0.35f);
    const float albedo = 0.34f + (checker ? 0.05f : -0.02f) * contrast;

    const float shadow = expf(-opticalDepthToLight(density, p, params, 1.5f));
    float occlusion = 1.0f;
    if (p.x > 0.0f && p.z > 0.0f && p.x < params.bounds.x && p.z < params.bounds.z) {
        const float invCellSize = 1.0f / params.cellSize;
        const float2 depth = tex3D<float2>(lightDepth,
            p.x * invCellSize, 0.5f, p.z * invCellSize);
        occlusion = expf(-0.35f * depth.y);
    }

    const float sunFactor = fmaxf(params.lightDirection.y, 0.0f) * shadow;
    const float3 lit = add(
        multiply(params.sunRadiance, sunFactor),
        multiply(params.skyRadiance, occlusion));
    const float3 color = multiply(lit, albedo);

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

__global__ void renderKernel(
    cudaTextureObject_t density,
    cudaTextureObject_t temperature,
    cudaTextureObject_t lightDepth,
    RenderParams params,
    RenderCamera camera,
    int width,
    int height,
    uint8_t* pixels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float ndcX = 2.0f * (static_cast<float>(x) + 0.5f) / width - 1.0f;
    const float ndcY = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / height;
    float3 origin = camera.position;
    float3 direction;
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

    const float jitter = gradientNoise(static_cast<float>(x), static_cast<float>(y));
    const float invCellSize = 1.0f / params.cellSize;
    float floorDistance = FLT_MAX;
    if (params.lit && params.floor && direction.y < -1e-6f && origin.y > 0.0f) {
        floorDistance = -origin.y / direction.y;
    }

    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    float entry;
    float exit;
    if (intersectVolume(origin, direction, params.bounds, entry, exit)) {
        exit = fminf(exit, floorDistance);
        const float stepSize = params.cellSize * 0.5f;
        const int maxSteps = static_cast<int>(ceilf((exit - entry) / stepSize));
        float distance = entry + stepSize * jitter;

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
                radiance = add(radiance, multiply(make_float3(0.92f, 0.96f, 1.0f), contribution));
                transmittance *= 1.0f - alpha;
                continue;
            }

            const float2 depth = tex3D<float2>(lightDepth,
                p.x * invCellSize, p.y * invCellSize, p.z * invCellSize);
            float sun = 0.0f;
            float weight = 1.0f;
            float extinctionScale = 1.0f;
            for (int octave = 0; octave < 3; ++octave) {
                sun += weight * phases[octave] * expf(-depth.x * extinctionScale);
                weight *= 0.3f;
                extinctionScale *= 0.5f;
            }
            const float skyVisibility = expf(-depth.y);
            const float3 inScattered = multiply(add(
                multiply(params.sunRadiance, sun),
                multiply(params.skyRadiance, skyVisibility)),
                params.albedo);

            // Analytic integration of constant in-scattering over the step.
            const float stepTransmittance = expf(-value * params.extinction * stepSize);
            radiance = add(radiance, multiply(
                add(multiply(inScattered, 1.0f - stepTransmittance), emitted), transmittance));
            transmittance *= stepTransmittance;
        }
    }

    if (params.lit && transmittance > 0.003f) {
        float3 background;
        if (floorDistance < FLT_MAX) {
            const float3 p = add(origin, multiply(direction, floorDistance));
            background = floorColor(p, direction, floorDistance, density, lightDepth, params);
        } else {
            background = skyColor(direction, params);
        }
        radiance = add(radiance, multiply(background, transmittance));
    }

    const float3 color = params.lit ? toneMap(radiance, params.exposure) : radiance;
    const float dither = (gradientNoise(x + 17.0f, y + 59.0f) - 0.5f) / 255.0f;
    const int pixel = (y * width + x) * 4;
    pixels[pixel + 0] = static_cast<uint8_t>(saturate(color.x + dither) * 255.0f + 0.5f);
    pixels[pixel + 1] = static_cast<uint8_t>(saturate(color.y + dither) * 255.0f + 0.5f);
    pixels[pixel + 2] = static_cast<uint8_t>(saturate(color.z + dither) * 255.0f + 0.5f);
    pixels[pixel + 3] = 255;
}

float3 toFloat3(const Vector3& value) {
    return make_float3(value.x, value.y, value.z);
}

}

VolumeTexture::VolumeTexture(int nx, int ny, int nz, cudaChannelFormatDesc format, bool writable)
    : nx_(nx), ny_(ny), nz_(nz) {
    CUDA_CHECK(cudaMalloc3DArray(&array_, &format, make_cudaExtent(nx, ny, nz),
        writable ? cudaArraySurfaceLoadStore : cudaArrayDefault));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array_;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeBorder;
    description.addressMode[1] = cudaAddressModeBorder;
    description.addressMode[2] = cudaAddressModeBorder;
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
      lightDepth_(solver.nx(), solver.ny(), solver.nz(), cudaCreateChannelDesc<float2>(), true),
      skyDepth_(checkedElementCount(solver.nx(), solver.ny(), solver.nz())) {
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
    devicePixels_ = std::move(pixels);
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
    const GridView3D<const float> density = solver_.densityView();
    density_.copyFrom(density);
    if (lighting_.emission) {
        temperature_.copyFrom(solver_.temperatureView());
    }
    if (!lighting_.lit) {
        return;
    }

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
        density_.texture(), skyDepth_.data(), lightDepth_.surface(),
        density.nx, density.ny, density.nz, params);
    CUDA_CHECK(cudaGetLastError());
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

    TIME(updateVolumes());

    constexpr int tile = 16;
    const dim3 threads(tile, tile);
    const dim3 blocks(
        (static_cast<unsigned int>(width_) + tile - 1) / tile,
        (static_cast<unsigned int>(height_) + tile - 1) / tile);
    renderKernel<<<blocks, threads>>>(
        density_.texture(), temperature_.texture(), lightDepth_.texture(),
        makeRenderParams(solver_, lighting_), renderCamera,
        static_cast<int>(width_), static_cast<int>(height_), devicePixels_.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        destination, devicePixels_.data(), pixelBytes, cudaMemcpyDeviceToHost));
}
