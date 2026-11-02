#pragma once

#include "cuda/cuda_fluid_solver.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

// Runtime frame profiler. CPU stages are timed with a steady clock; GPU stages are bracketed
// with CUDA events on the default stream and resolved once per frame, so timing a stage does
// not synchronize the device. While disabled, every scope reduces to a single branch.

enum class CpuStage {
    Input,
    Simulation,   // Submitting all simulation steps of the frame
    Render,       // CUDA rendering or divergence slice, including the pixel readback wait
    Upload,       // SDL_UpdateTexture
    Ui,
    Present,
    Count
};

enum class GpuStage {
    AdvectVelocity,
    Sources,
    Forces,
    Boundary,
    Divergence,
    Pressure,
    Projection,
    AdvectScalars,
    Diagnostics,
    Volumes,      // Density / temperature upload and light volume
    RayMarch,
    Readback,
    Count
};

constexpr int kCpuStageCount = static_cast<int>(CpuStage::Count);
constexpr int kGpuStageCount = static_cast<int>(GpuStage::Count);

struct StageInfo {
    const char* label;  // Shown in the UI
    const char* key;    // CSV column name
};

const StageInfo& stageInfo(CpuStage stage);
const StageInfo& stageInfo(GpuStage stage);

// Stages up to and including Diagnostics belong to CudaFluidSolver::step.
constexpr bool isSimulationStage(GpuStage stage) {
    return static_cast<int>(stage) <= static_cast<int>(GpuStage::Diagnostics);
}

struct FrameRecord {
    uint64_t frame = 0;
    double time = 0.0;        // Seconds since the profiler was created, at the end of the frame
    float frameMs = 0.0f;     // Wall-clock time since the previous recorded frame
    int simSteps = 0;
    std::array<float, kCpuStageCount> cpuMs{};
    std::array<float, kGpuStageCount> gpuMs{};  // Summed over all simulation steps of the frame
    bool hasDiagnostics = false;
    SolverDiagnostics diagnostics;

    float gpuSimulationMs() const;
    float gpuRenderMs() const;
};

class FrameProfiler {
public:
    static constexpr double kMaxHistorySeconds = 60.0;
    static constexpr size_t kMaxHistoryFrames = size_t(1) << 17;

    FrameProfiler();
    ~FrameProfiler();

    FrameProfiler(const FrameProfiler&) = delete;
    FrameProfiler& operator=(const FrameProfiler&) = delete;

    // Collection is on while the window is shown or a recording is running.
    // Changes take effect at the next beginFrame so that a frame is never half measured.
    bool enabled() const noexcept { return enabled_; }
    bool windowVisible() const noexcept { return windowVisible_; }
    void setWindowVisible(bool visible) noexcept { windowVisible_ = visible; }

    void beginFrame();
    void endFrame(int simSteps);

    // Diagnostics are sampled on one step every `interval` frames; 0 turns them off.
    int diagnosticsInterval() const noexcept { return diagnosticsInterval_; }
    void setDiagnosticsInterval(int interval) noexcept { diagnosticsInterval_ = interval < 0 ? 0 : interval; }
    bool wantsDiagnostics() const noexcept;
    void setDiagnostics(const SolverDiagnostics& diagnostics);

    void beginCpu(CpuStage stage) noexcept;
    void endCpu(CpuStage stage) noexcept;
    int beginGpu(GpuStage stage);
    void endGpu(int timer) noexcept;

    const std::deque<FrameRecord>& history() const noexcept { return history_; }
    void clearHistory() { history_.clear(); }

    // "# key: value" lines written at the top of every CSV file.
    void setMetadata(std::vector<std::pair<std::string, std::string>> metadata) { metadata_ = std::move(metadata); }

    // Streams every subsequent frame to a CSV file.
    bool startRecording(const std::string& path);
    void stopRecording();
    bool recording() const noexcept { return recording_.is_open(); }
    const std::string& recordingPath() const noexcept { return recordingPath_; }
    uint64_t recordedFrames() const noexcept { return recordedFrames_; }

    // Writes the frames of the last `windowSeconds` of history to a CSV file.
    bool exportHistory(const std::string& path, double windowSeconds) const;

    // Destroys the CUDA events. Call before the CUDA context is torn down.
    void releaseGpuResources() noexcept;

private:
    using Clock = std::chrono::steady_clock;

    struct GpuTimer {
        GpuStage stage;
        cudaEvent_t start;
        cudaEvent_t stop;
    };

    void resolveGpuTimers();
    void writeCsvHeader(std::ostream& out) const;
    static void writeCsvRow(std::ostream& out, const FrameRecord& record);
    double secondsSinceStart(Clock::time_point t) const;

    bool enabled_ = false;
    bool windowVisible_ = false;
    int diagnosticsInterval_ = 1;

    Clock::time_point start_;
    Clock::time_point frameStart_;
    Clock::time_point lastFrameEnd_;
    bool hasLastFrame_ = false;
    std::array<Clock::time_point, kCpuStageCount> cpuStarts_{};

    std::vector<GpuTimer> gpuTimers_;
    size_t usedGpuTimers_ = 0;

    uint64_t frameIndex_ = 0;
    FrameRecord current_;
    std::deque<FrameRecord> history_;

    std::vector<std::pair<std::string, std::string>> metadata_;
    std::ofstream recording_;
    std::string recordingPath_;
    uint64_t recordedFrames_ = 0;
};

FrameProfiler& profiler();

class CpuScope {
public:
    explicit CpuScope(CpuStage stage) noexcept : stage_(stage) { profiler().beginCpu(stage_); }
    ~CpuScope() { profiler().endCpu(stage_); }
    CpuScope(const CpuScope&) = delete;
    CpuScope& operator=(const CpuScope&) = delete;

private:
    CpuStage stage_;
};

class GpuScope {
public:
    explicit GpuScope(GpuStage stage) : timer_(profiler().beginGpu(stage)) {}
    ~GpuScope() {
        if (timer_ >= 0) {
            profiler().endGpu(timer_);
        }
    }
    GpuScope(const GpuScope&) = delete;
    GpuScope& operator=(const GpuScope&) = delete;

private:
    int timer_;
};

// Time a single call. GPU stages must not nest, otherwise their time is counted twice.
#define PROFILE_CPU(stage, call) do { const CpuScope profile_scope_(stage); call; } while (false)
#define PROFILE_GPU(stage, call) do { const GpuScope profile_scope_(stage); call; } while (false)
