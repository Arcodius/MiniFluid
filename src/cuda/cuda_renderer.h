#pragma once

#include "grid.h"

#include <cstddef>
#include <cstdint>
#include <utility>

class Camera;
class CudaFluidSolver;

// A 3D CUDA array with a hardware-filtered texture object and an optional surface object.
class VolumeTexture {
public:
    VolumeTexture() = default;
    VolumeTexture(int nx, int ny, int nz, cudaChannelFormatDesc format, bool writable);
    ~VolumeTexture() noexcept;

    VolumeTexture(const VolumeTexture&) = delete;
    VolumeTexture& operator=(const VolumeTexture&) = delete;
    VolumeTexture(VolumeTexture&& other) noexcept;
    VolumeTexture& operator=(VolumeTexture&& other) noexcept;

    cudaTextureObject_t texture() const noexcept { return texture_; }
    cudaSurfaceObject_t surface() const noexcept { return surface_; }

    // Copies a linear device grid with matching dimensions into the array.
    void copyFrom(GridView3D<const float> source);

private:
    void releaseNoexcept() noexcept;

    cudaArray_t array_ = nullptr;
    cudaTextureObject_t texture_ = 0;
    cudaSurfaceObject_t surface_ = 0;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
};

struct LightingSettings {
    float lightAzimuth = 320.0f;   // Degrees around the vertical axis
    float lightElevation = 38.0f;  // Degrees above the horizon
    float3 sunColor = make_float3(1.0f, 0.93f, 0.82f);
    float sunIntensity = 3.2f;
    float3 skyColor = make_float3(0.52f, 0.66f, 0.92f);
    float ambientIntensity = 0.3f;
    float extinction = 90.0f;      // Extinction per world unit at unit smoke density
    float albedo = 0.85f;
    float phaseG = 0.5f;           // Henyey-Greenstein anisotropy of the forward lobe
    float emissionStrength = 180.0f;
    float exposure = 0.6f;
    bool lit = true;               // false: unlit opacity accumulation on black
    bool emission = false;         // Temperature-driven glow
    bool floor = true;
};

class CudaRenderer {
private:
    size_t width_ = 0;
    size_t height_ = 0;
    Camera& camera_;
    CudaFluidSolver& solver_;
    DeviceBuffer<uint8_t> devicePixels_;

    VolumeTexture density_;
    VolumeTexture temperature_;
    VolumeTexture lightDepth_;     // float2: optical depth toward the sun, and toward the sky
    DeviceBuffer<float> skyDepth_;
    LightingSettings lighting_;

    void updateVolumes();

public:
    explicit CudaRenderer(CudaFluidSolver& solver, Camera& camera, std::pair<size_t, size_t> resolution = {720, 720});

    void render(uint8_t* destination, size_t destinationSize);
    void setResolution(size_t w, size_t h);
    std::pair<size_t, size_t> getResolution() const;

    LightingSettings& lighting() noexcept { return lighting_; }
    const LightingSettings& lighting() const noexcept { return lighting_; }
};
