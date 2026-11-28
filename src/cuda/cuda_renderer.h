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
    // clampToEdge: repeat edge texels outside the grid instead of returning zero (needed for distance fields).
    VolumeTexture(int nx, int ny, int nz, cudaChannelFormatDesc format, bool writable,
        bool clampToEdge = false);
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

// Quality settings of the ray march. Changing either restarts the history.
struct RenderSettings {
    float stepCells = 1.0f;  // Ray march step length, in grid cells
    // The ray march runs at the output resolution divided by this (rounded up). An integer, so render pixels tile the output evenly.
    int renderScale = 1;

    bool operator==(const RenderSettings& other) const {
        return (stepCells == other.stepCells) && (renderScale == other.renderScale);
    }

    bool operator!=(const RenderSettings& other) const {
        return !(*this == other);
    }
};

// The camera as the ray generation sees it: a plain copy of the values one frame was rendered with.
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

class CudaRenderer {
private:
    // Two resolutions: the output (what render() returns, and the history) and the ray march's,
    // which is smaller by RenderSettings::renderScale.
    size_t width_ = 0;
    size_t height_ = 0;
    size_t renderWidth_ = 0;
    size_t renderHeight_ = 0;
    Camera& camera_;
    CudaFluidSolver& solver_;
    // Output resolution
    DeviceBuffer<uint8_t> devicePixels_;
    DeviceBuffer<float4> historyHdrPixels0_;
    DeviceBuffer<float4> historyHdrPixels1_;
    // Render resolution
    DeviceBuffer<float4> hdrPixels_;
    DeviceBuffer<float3> motionPixels_;

    VolumeTexture density_;
    VolumeTexture temperature_;
    VolumeTexture lightDepth_;     // float4: smoke optical depth toward the sun and the sky, then solid visibility toward the sun and the sky
    VolumeTexture sdf_;            // Signed distance to the solid, clamped at the domain edge
    VolumeTexture velocity_; // Averaged velocity
    DeviceBuffer<float> skyDepth_;
    LightingSettings lighting_;
    RenderSettings renderSettings_;

    int frameIndex_;
    float lastRenderTime_ = 0.0f;

    // Temporal accumulation. The history is restarted whenever what it averages changes: the camera, the simulation state, or a lighting setting that affects the HDR radiance.
    static constexpr int kMaxAccumulatedFrames = 128;
    uint64_t historySolverVersion_ = 0;
    LightingSettings historyLighting_;
    RenderSettings historyRenderSettings_;
    // The camera the previous frame was rendered with, for reprojection. A value, not a reference:
    // it must keep the old state while camera_ moves on. Invalid until a frame has been rendered.
    RenderCamera previousCamera_{};
    bool previousCameraValid_ = false;

    // What each cached volume was last built from, so updateVolumes only redoes the work whose inputs changed. kNeverBuilt matches no solver version, so the first frame builds everything.
    static constexpr uint64_t kNeverBuilt = UINT64_MAX;
    uint64_t densityVersion_ = kNeverBuilt;      // density_ and sdf_
    uint64_t temperatureVersion_ = kNeverBuilt;  // temperature_, uploaded only while emission is on
    // The light volume (skyDepth_ and lightDepth_) depends on the smoke and the solid, the sun direction, and the extinction. The other lighting settings are only read per pixel.
    struct LightVolumeInputs {
        uint64_t solverVersion = kNeverBuilt;
        float lightAzimuth = 0.0f;
        float lightElevation = 0.0f;
        float extinction = 0.0f;

        bool operator==(const LightVolumeInputs& other) const {
            return solverVersion == other.solverVersion && lightAzimuth == other.lightAzimuth
                && lightElevation == other.lightElevation && extinction == other.extinction;
        }
    };
    LightVolumeInputs lightVolumeInputs_;

    void renderToDevice();

    void updateVolumes();
    void updateRenderResolution();  // Reallocates the render-resolution buffers when the scale changed
    bool historyNeedsReset();

public:
    explicit CudaRenderer(CudaFluidSolver& solver, Camera& camera, std::pair<size_t, size_t> resolution = {720, 720});

    // Old version: render + read back to frameBuffer
    void render(uint8_t* destination, size_t destinationSize);
    // New version: render + D2D copy to texture
    void render(cudaArray_t destination);
    void setResolution(size_t w, size_t h);
    std::pair<size_t, size_t> getResolution() const;

    LightingSettings& lighting() noexcept { return lighting_; }
    const LightingSettings& lighting() const noexcept { return lighting_; }
    // Read every frame by render(), so the UI can edit it directly, like lighting().
    RenderSettings& renderSettings() noexcept { return renderSettings_; }
    const RenderSettings& renderSettings() const noexcept { return renderSettings_; }
};
