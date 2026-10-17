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

__device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 multiply(float3 value, float scale) {
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__device__ float3 normalize(float3 value) {
    const float inverseLength = rsqrtf(
        value.x * value.x + value.y * value.y + value.z * value.z);
    return multiply(value, inverseLength);
}

__device__ float sampleDensity(
    GridView3D<const float> density,
    float x,
    float y,
    float z,
    float cellSize) {
    const float gridX = x / cellSize;
    const float gridY = y / cellSize;
    const float gridZ = z / cellSize;
    if (gridX < 0.0f || gridY < 0.0f || gridZ < 0.0f
        || gridX >= density.nx || gridY >= density.ny || gridZ >= density.nz) {
        return 0.0f;
    }

    const float cellX = gridX - 0.5f;
    const float cellY = gridY - 0.5f;
    const float cellZ = gridZ - 0.5f;
    const int x0 = max(static_cast<int>(floorf(cellX)), 0);
    const int y0 = max(static_cast<int>(floorf(cellY)), 0);
    const int z0 = max(static_cast<int>(floorf(cellZ)), 0);
    const int x1 = min(x0 + 1, density.nx - 1);
    const int y1 = min(y0 + 1, density.ny - 1);
    const int z1 = min(z0 + 1, density.nz - 1);
    const float tx = fminf(fmaxf(cellX, 0.0f) - x0, 1.0f);
    const float ty = fminf(fmaxf(cellY, 0.0f) - y0, 1.0f);
    const float tz = fminf(fmaxf(cellZ, 0.0f) - z0, 1.0f);

    const float x00 = density(x0, y0, z0)
        + tx * (density(x1, y0, z0) - density(x0, y0, z0));
    const float x10 = density(x0, y1, z0)
        + tx * (density(x1, y1, z0) - density(x0, y1, z0));
    const float x01 = density(x0, y0, z1)
        + tx * (density(x1, y0, z1) - density(x0, y0, z1));
    const float x11 = density(x0, y1, z1)
        + tx * (density(x1, y1, z1) - density(x0, y1, z1));
    const float y0Value = x00 + ty * (x10 - x00);
    const float y1Value = x01 + ty * (x11 - x01);
    return fmaxf(y0Value + tz * (y1Value - y0Value), 0.0f);
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

__global__ void renderKernel(
    GridView3D<const float> density,
    float cellSize,
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

    const float3 bounds = make_float3(
        density.nx * cellSize,
        density.ny * cellSize,
        density.nz * cellSize);
    float entry;
    float exit;
    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;
    float transmittance = 1.0f;
    if (intersectVolume(origin, direction, bounds, entry, exit)) {
        const float stepSize = cellSize * 0.5f;
        const float3 step = multiply(direction, stepSize);
        const int maxSteps = static_cast<int>(ceilf((exit - entry) / stepSize));
        float distance = entry + stepSize * 0.5f;
        float3 samplePosition = add(origin, multiply(direction, distance));

        for (int sample = 0; sample < maxSteps && transmittance > 0.01f;
             ++sample, distance += stepSize, samplePosition = add(samplePosition, step)) {
            const float value = sampleDensity(
                 density, samplePosition.x, samplePosition.y,
                 samplePosition.z, cellSize);
            const float alpha = 1.0f - expf(-value * 5.0f * stepSize);
            const float contribution = transmittance * alpha;
            red += contribution * 0.92f;
            green += contribution * 0.96f;
            blue += contribution;
            transmittance *= 1.0f - alpha;
        }
    }

    const int pixel = (y * width + x) * 4;
    pixels[pixel + 0] = static_cast<uint8_t>(fminf(red, 1.0f) * 255.0f + 0.5f);
    pixels[pixel + 1] = static_cast<uint8_t>(fminf(green, 1.0f) * 255.0f + 0.5f);
    pixels[pixel + 2] = static_cast<uint8_t>(fminf(blue, 1.0f) * 255.0f + 0.5f);
    pixels[pixel + 3] = 255;
}

float3 toFloat3(const Vector3& value) {
    return make_float3(value.x, value.y, value.z);
}

}

CudaRenderer::CudaRenderer(
    CudaFluidSolver& solver,
    Camera& camera,
    std::pair<size_t, size_t> resolution)
    : camera_(camera), solver_(solver) {
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

    constexpr int tile = 16;
    const dim3 threads(tile, tile);
    const dim3 blocks(
        (static_cast<unsigned int>(width_) + tile - 1) / tile,
        (static_cast<unsigned int>(height_) + tile - 1) / tile);
    renderKernel<<<blocks, threads>>>(
        solver_.densityView(), solver_.spacing(), renderCamera,
        static_cast<int>(width_), static_cast<int>(height_), devicePixels_.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        destination, devicePixels_.data(), pixelBytes, cudaMemcpyDeviceToHost));
}