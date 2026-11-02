#include "profiler.h"

#include "cuda/cuda_check.h"

#include <iomanip>

namespace {

constexpr std::array<StageInfo, kCpuStageCount> kCpuStages = {{
    {"Input", "cpu_input_ms"},
    {"Simulation", "cpu_simulation_ms"},
    {"Render", "cpu_render_ms"},
    {"Texture upload", "cpu_upload_ms"},
    {"UI", "cpu_ui_ms"},
    {"Present", "cpu_present_ms"},
}};

constexpr std::array<StageInfo, kGpuStageCount> kGpuStages = {{
    {"Advect velocity", "gpu_advect_velocity_ms"},
    {"Sources", "gpu_sources_ms"},
    {"Forces", "gpu_forces_ms"},
    {"Boundary", "gpu_boundary_ms"},
    {"Divergence", "gpu_divergence_ms"},
    {"Pressure solve", "gpu_pressure_ms"},
    {"Projection", "gpu_projection_ms"},
    {"Advect scalars", "gpu_advect_scalars_ms"},
    {"Diagnostics", "gpu_diagnostics_ms"},
    {"Volumes + light", "gpu_volumes_ms"},
    {"Ray march", "gpu_ray_march_ms"},
    {"Readback", "gpu_readback_ms"},
}};

float millisecondsBetween(std::chrono::steady_clock::time_point from,
                          std::chrono::steady_clock::time_point to) {
    return std::chrono::duration<float, std::milli>(to - from).count();
}

} // namespace

const StageInfo& stageInfo(CpuStage stage) {
    return kCpuStages[static_cast<size_t>(stage)];
}

const StageInfo& stageInfo(GpuStage stage) {
    return kGpuStages[static_cast<size_t>(stage)];
}

float FrameRecord::gpuSimulationMs() const {
    float sum = 0.0f;
    for (int s = 0; s < kGpuStageCount; ++s) {
        if (isSimulationStage(static_cast<GpuStage>(s))) {
            sum += gpuMs[s];
        }
    }
    return sum;
}

float FrameRecord::gpuRenderMs() const {
    float sum = 0.0f;
    for (int s = 0; s < kGpuStageCount; ++s) {
        if (!isSimulationStage(static_cast<GpuStage>(s))) {
            sum += gpuMs[s];
        }
    }
    return sum;
}

FrameProfiler& profiler() {
    static FrameProfiler instance;
    return instance;
}

FrameProfiler::FrameProfiler() : start_(Clock::now()) {}

FrameProfiler::~FrameProfiler() {
    releaseGpuResources();
}

void FrameProfiler::releaseGpuResources() noexcept {
    for (const GpuTimer& timer : gpuTimers_) {
        cudaEventDestroy(timer.start);
        cudaEventDestroy(timer.stop);
    }
    gpuTimers_.clear();
    usedGpuTimers_ = 0;
}

double FrameProfiler::secondsSinceStart(Clock::time_point t) const {
    return std::chrono::duration<double>(t - start_).count();
}

void FrameProfiler::beginFrame() {
    const bool wasEnabled = enabled_;
    enabled_ = windowVisible_ || recording();
    if (enabled_ && !wasEnabled) {
        // Do not count the time spent with the profiler off as one long frame.
        hasLastFrame_ = false;
    }
    frameStart_ = Clock::now();
    current_ = FrameRecord{};
}

void FrameProfiler::endFrame(int simSteps) {
    if (!enabled_) {
        return;
    }
    resolveGpuTimers();

    const Clock::time_point now = Clock::now();
    current_.frame = frameIndex_++;
    current_.time = secondsSinceStart(now);
    current_.frameMs = millisecondsBetween(hasLastFrame_ ? lastFrameEnd_ : frameStart_, now);
    current_.simSteps = simSteps;
    lastFrameEnd_ = now;
    hasLastFrame_ = true;

    history_.push_back(current_);
    while (history_.size() > kMaxHistoryFrames
           || history_.front().time < current_.time - kMaxHistorySeconds) {
        history_.pop_front();
    }

    if (recording_.is_open()) {
        writeCsvRow(recording_, current_);
        ++recordedFrames_;
    }
}

bool FrameProfiler::wantsDiagnostics() const noexcept {
    return enabled_ && diagnosticsInterval_ > 0 && frameIndex_ % diagnosticsInterval_ == 0;
}

void FrameProfiler::setDiagnostics(const SolverDiagnostics& diagnostics) {
    if (!enabled_) {
        return;
    }
    current_.hasDiagnostics = true;
    current_.diagnostics = diagnostics;
}

void FrameProfiler::beginCpu(CpuStage stage) noexcept {
    if (enabled_) {
        cpuStarts_[static_cast<size_t>(stage)] = Clock::now();
    }
}

void FrameProfiler::endCpu(CpuStage stage) noexcept {
    if (enabled_) {
        const size_t index = static_cast<size_t>(stage);
        current_.cpuMs[index] += millisecondsBetween(cpuStarts_[index], Clock::now());
    }
}

int FrameProfiler::beginGpu(GpuStage stage) {
    if (!enabled_) {
        return -1;
    }
    if (usedGpuTimers_ == gpuTimers_.size()) {
        GpuTimer timer{stage, nullptr, nullptr};
        CUDA_CHECK(cudaEventCreate(&timer.start));
        CUDA_CHECK(cudaEventCreate(&timer.stop));
        gpuTimers_.push_back(timer);
    }
    GpuTimer& timer = gpuTimers_[usedGpuTimers_];
    timer.stage = stage;
    CUDA_CHECK(cudaEventRecord(timer.start, 0));
    return static_cast<int>(usedGpuTimers_++);
}

void FrameProfiler::endGpu(int timer) noexcept {
    // A failed record surfaces as an error when the timer is resolved.
    cudaEventRecord(gpuTimers_[static_cast<size_t>(timer)].stop, 0);
}

void FrameProfiler::resolveGpuTimers() {
    for (size_t i = 0; i < usedGpuTimers_; ++i) {
        const GpuTimer& timer = gpuTimers_[i];
        CUDA_CHECK(cudaEventSynchronize(timer.stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, timer.start, timer.stop));
        current_.gpuMs[static_cast<size_t>(timer.stage)] += ms;
    }
    usedGpuTimers_ = 0;
}

bool FrameProfiler::startRecording(const std::string& path) {
    stopRecording();
    recording_.open(path, std::ios::out | std::ios::trunc);
    if (!recording_.is_open()) {
        return false;
    }
    recordingPath_ = path;
    recordedFrames_ = 0;
    writeCsvHeader(recording_);
    return true;
}

void FrameProfiler::stopRecording() {
    if (recording_.is_open()) {
        recording_.close();
    }
}

bool FrameProfiler::exportHistory(const std::string& path, double windowSeconds) const {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    writeCsvHeader(out);
    const double from = history_.empty() ? 0.0 : history_.back().time - windowSeconds;
    for (const FrameRecord& record : history_) {
        if (record.time >= from) {
            writeCsvRow(out, record);
        }
    }
    return static_cast<bool>(out);
}

void FrameProfiler::writeCsvHeader(std::ostream& out) const {
    out << "# MiniFluid profile\n";
    for (const auto& [key, value] : metadata_) {
        out << "# " << key << ": " << value << '\n';
    }
    out << "frame,time_s,frame_ms,sim_steps";
    for (const StageInfo& stage : kCpuStages) {
        out << ',' << stage.key;
    }
    for (const StageInfo& stage : kGpuStages) {
        out << ',' << stage.key;
    }
    out << ",gpu_sim_ms,gpu_render_ms"
           ",sim_time_s,div_before_max,div_before_rms,div_after_max,div_after_rms"
           ",pressure_residual,max_speed,cfl,kinetic_energy,smoke_mass,max_density,max_temperature\n";
}

void FrameProfiler::writeCsvRow(std::ostream& out, const FrameRecord& r) {
    out << r.frame << ',' << std::fixed << std::setprecision(6) << r.time
        << std::defaultfloat << std::setprecision(6)
        << ',' << r.frameMs << ',' << r.simSteps;
    for (float ms : r.cpuMs) {
        out << ',' << ms;
    }
    for (float ms : r.gpuMs) {
        out << ',' << ms;
    }
    out << ',' << r.gpuSimulationMs() << ',' << r.gpuRenderMs();
    if (r.hasDiagnostics) {
        const SolverDiagnostics& d = r.diagnostics;
        out << ',' << d.time
            << ',' << d.divergenceBeforeMax << ',' << d.divergenceBeforeRms
            << ',' << d.divergenceAfterMax << ',' << d.divergenceAfterRms
            << ',' << d.pressureResidual << ',' << d.maxSpeed << ',' << d.cfl
            << ',' << d.kineticEnergy << ',' << d.smokeMass
            << ',' << d.maxDensity << ',' << d.maxTemperature;
    } else {
        out << ",,,,,,,,,,,,";
    }
    out << '\n';
}
