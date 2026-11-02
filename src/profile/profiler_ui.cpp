#include "profiler_ui.h"

#include "profiler.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>

namespace {

constexpr double kRefreshSeconds = 0.1;  // Statistics and plots are rebuilt at this rate
constexpr int kPlotBuckets = 240;
constexpr double kWindowChoices[] = {5.0, 10.0, 30.0, 60.0};
constexpr const char* kWindowLabels[] = {"5 s", "10 s", "30 s", "60 s"};

struct WindowSummary {
    int frames = 0;
    double seconds = 0.0;       // Sum of frame times
    float fpsAvg = 0.0f;
    float low1Fps = 0.0f;       // FPS at the 99th percentile frame time
    float frameMsAvg = 0.0f;
    float frameMsMin = 0.0f;
    float frameMsMax = 0.0f;
    float frameMsP99 = 0.0f;

    int steps = 0;
    float stepsPerFrame = 0.0f;
    float realtimeFactor = 0.0f;  // Simulated seconds per wall-clock second
    float gpuStepMs = 0.0f;       // Simulation stages per step, without diagnostics
    float mcellsPerSecond = 0.0f;

    std::array<float, kCpuStageCount> cpuAvg{};
    std::array<float, kCpuStageCount> cpuMax{};
    std::array<float, kGpuStageCount> gpuAvg{};
    std::array<float, kGpuStageCount> gpuMax{};
    float gpuSimAvg = 0.0f;
    float gpuSimMax = 0.0f;
    float gpuRenderAvg = 0.0f;
    float gpuRenderMax = 0.0f;

    std::vector<float> fpsPlot;        // Frames per second in each time bucket
    std::vector<float> frameMsPlot;    // Slowest frame in each time bucket
    std::vector<float> divergencePlot; // log10 max |div| after projection
    std::vector<float> residualPlot;   // log10 relative pressure residual
    std::vector<float> cflPlot;

    bool hasDiagnostics = false;
    SolverDiagnostics latest;
};

struct UiState {
    int windowChoice = 1;
    double lastRefresh = -1.0;
    uint64_t lastFrame = ~uint64_t(0);
    WindowSummary summary;
    std::string message;
    bool messageIsError = false;

    std::string gpuName;
    double lastMemoryQuery = -1.0;
    size_t memoryFree = 0;
    size_t memoryTotal = 0;
};

UiState& uiState() {
    static UiState state;
    return state;
}

float safeLog10(float value) {
    return std::log10(std::max(value, 1.0e-12f));
}

// Replaces empty buckets with the previous value so that sparse samples plot as steps.
void fillGaps(std::vector<float>& values, const std::vector<bool>& filled) {
    float previous = 0.0f;
    size_t first = 0;
    while (first < values.size() && !filled[first]) {
        ++first;
    }
    if (first < values.size()) {
        previous = values[first];
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (filled[i]) {
            previous = values[i];
        } else {
            values[i] = previous;
        }
    }
}

WindowSummary summarize(const std::deque<FrameRecord>& history, double window,
                        const ProfilerContext& context) {
    WindowSummary s;
    if (history.empty()) {
        return s;
    }
    const double newest = history.back().time;
    const double from = std::max(newest - window, history.front().time);
    const double bucketWidth = std::max(newest - from, 1.0e-6) / kPlotBuckets;

    s.fpsPlot.assign(kPlotBuckets, 0.0f);
    s.frameMsPlot.assign(kPlotBuckets, 0.0f);
    s.divergencePlot.assign(kPlotBuckets, 0.0f);
    s.residualPlot.assign(kPlotBuckets, 0.0f);
    s.cflPlot.assign(kPlotBuckets, 0.0f);
    std::vector<int> bucketFrames(kPlotBuckets, 0);
    std::vector<bool> bucketDiagnostics(kPlotBuckets, false);

    std::vector<float> frameTimes;
    double frameMsSum = 0.0;
    std::array<double, kCpuStageCount> cpuSum{};
    std::array<double, kGpuStageCount> gpuSum{};
    double gpuSimSum = 0.0;
    double gpuRenderSum = 0.0;
    s.frameMsMin = FLT_MAX;

    for (const FrameRecord& r : history) {
        if (r.time < from) {
            continue;
        }
        ++s.frames;
        frameTimes.push_back(r.frameMs);
        frameMsSum += r.frameMs;
        s.frameMsMin = std::min(s.frameMsMin, r.frameMs);
        s.frameMsMax = std::max(s.frameMsMax, r.frameMs);
        s.steps += r.simSteps;
        for (int i = 0; i < kCpuStageCount; ++i) {
            cpuSum[i] += r.cpuMs[i];
            s.cpuMax[i] = std::max(s.cpuMax[i], r.cpuMs[i]);
        }
        for (int i = 0; i < kGpuStageCount; ++i) {
            gpuSum[i] += r.gpuMs[i];
            s.gpuMax[i] = std::max(s.gpuMax[i], r.gpuMs[i]);
        }
        const float sim = r.gpuSimulationMs();
        const float render = r.gpuRenderMs();
        gpuSimSum += sim;
        gpuRenderSum += render;
        s.gpuSimMax = std::max(s.gpuSimMax, sim);
        s.gpuRenderMax = std::max(s.gpuRenderMax, render);

        const int bucket = std::clamp(
            static_cast<int>((r.time - from) / bucketWidth), 0, kPlotBuckets - 1);
        ++bucketFrames[bucket];
        s.frameMsPlot[bucket] = std::max(s.frameMsPlot[bucket], r.frameMs);
        if (r.hasDiagnostics) {
            const SolverDiagnostics& d = r.diagnostics;
            s.divergencePlot[bucket] = safeLog10(d.divergenceAfterMax);
            s.residualPlot[bucket] = safeLog10(d.pressureResidual);
            s.cflPlot[bucket] = d.cfl;
            bucketDiagnostics[bucket] = true;
            s.hasDiagnostics = true;
            s.latest = d;
        }
    }
    if (s.frames == 0) {
        return s;
    }

    for (int b = 0; b < kPlotBuckets; ++b) {
        s.fpsPlot[b] = static_cast<float>(bucketFrames[b] / bucketWidth);
    }
    fillGaps(s.divergencePlot, bucketDiagnostics);
    fillGaps(s.residualPlot, bucketDiagnostics);
    fillGaps(s.cflPlot, bucketDiagnostics);

    const size_t p99 = std::min(frameTimes.size() - 1, frameTimes.size() * 99 / 100);
    std::nth_element(frameTimes.begin(), frameTimes.begin() + p99, frameTimes.end());
    s.frameMsP99 = frameTimes[p99];

    const double n = s.frames;
    s.seconds = frameMsSum / 1000.0;
    s.frameMsAvg = static_cast<float>(frameMsSum / n);
    s.fpsAvg = frameMsSum > 0.0 ? static_cast<float>(1000.0 * n / frameMsSum) : 0.0f;
    s.low1Fps = s.frameMsP99 > 0.0f ? 1000.0f / s.frameMsP99 : 0.0f;
    for (int i = 0; i < kCpuStageCount; ++i) {
        s.cpuAvg[i] = static_cast<float>(cpuSum[i] / n);
    }
    for (int i = 0; i < kGpuStageCount; ++i) {
        s.gpuAvg[i] = static_cast<float>(gpuSum[i] / n);
    }
    s.gpuSimAvg = static_cast<float>(gpuSimSum / n);
    s.gpuRenderAvg = static_cast<float>(gpuRenderSum / n);

    s.stepsPerFrame = static_cast<float>(s.steps / n);
    if (s.seconds > 0.0) {
        s.realtimeFactor = static_cast<float>(s.steps * context.simulationDt / s.seconds);
    }
    if (s.steps > 0) {
        const double stepMs =
            (gpuSimSum - gpuSum[static_cast<size_t>(GpuStage::Diagnostics)]) / s.steps;
        s.gpuStepMs = static_cast<float>(stepMs);
        const double cells = static_cast<double>(context.nx) * context.ny * context.nz;
        if (stepMs > 0.0) {
            s.mcellsPerSecond = static_cast<float>(cells / (stepMs / 1000.0) / 1.0e6);
        }
    }
    return s;
}

void plot(const char* id, const std::vector<float>& values, const char* overlay,
          float minimum, float maximum, float height) {
    ImGui::PlotLines(id, values.data(), static_cast<int>(values.size()), 0, overlay,
                     minimum, maximum, ImVec2(-FLT_MIN, height));
}

void plotAutoRange(const char* id, const std::vector<float>& values, const char* overlay,
                   float height) {
    if (values.empty()) {
        return;
    }
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    const float margin = std::max(0.05f * (*hi - *lo), 1.0e-3f);
    plot(id, values, overlay, *lo - margin, *hi + margin, height);
}

void stageRow(const char* label, float avg, float max, float total, bool bold) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    if (bold) {
        ImGui::TextUnformatted(label);
    } else {
        ImGui::Text("  %s", label);
    }
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%.3f", avg);
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("%.3f", max);
    ImGui::TableSetColumnIndex(3);
    const float fraction = total > 0.0f ? avg / total : 0.0f;
    char overlay[16];
    std::snprintf(overlay, sizeof(overlay), "%.1f%%", 100.0f * fraction);
    ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), overlay);
}

bool beginStageTable(const char* id) {
    if (!ImGui::BeginTable(id, 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        return false;
    }
    ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthStretch, 2.2f);
    ImGui::TableSetupColumn("avg ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("max ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("share", ImGuiTableColumnFlags_WidthStretch, 1.6f);
    ImGui::TableHeadersRow();
    return true;
}

void metricRow(const char* label, const char* format, float value, bool warn = false) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(label);
    ImGui::TableSetColumnIndex(1);
    if (!std::isfinite(value)) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%g", value);
    } else if (warn) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), format, value);
    } else {
        ImGui::Text(format, value);
    }
}

void drawFrameRate(const WindowSummary& s) {
    ImGui::Text("FPS %.1f   1%% low %.1f   frames %d", s.fpsAvg, s.low1Fps, s.frames);
    ImGui::Text("Frame ms  avg %.3f  min %.3f  max %.3f  p99 %.3f",
                s.frameMsAvg, s.frameMsMin, s.frameMsMax, s.frameMsP99);
    char overlay[64];
    float fpsMax = 0.0f;
    for (float v : s.fpsPlot) {
        fpsMax = std::max(fpsMax, v);
    }
    std::snprintf(overlay, sizeof(overlay), "FPS (avg %.0f)", s.fpsAvg);
    plot("##fps", s.fpsPlot, overlay, 0.0f, std::max(fpsMax * 1.1f, 1.0f), 70.0f);
    std::snprintf(overlay, sizeof(overlay), "frame ms, worst per bucket (max %.2f)", s.frameMsMax);
    plot("##frame_ms", s.frameMsPlot, overlay, 0.0f, std::max(s.frameMsMax * 1.1f, 0.1f), 50.0f);

    ImGui::Text("Steps/frame %.2f   realtime x%.2f", s.stepsPerFrame, s.realtimeFactor);
    ImGui::Text("GPU step %.3f ms   %.1f Mcells/s", s.gpuStepMs, s.mcellsPerSecond);
}

void drawGpuStages(const WindowSummary& s) {
    if (!beginStageTable("gpu_stages")) {
        return;
    }
    const float total = s.gpuSimAvg + s.gpuRenderAvg;
    stageRow("Simulation", s.gpuSimAvg, s.gpuSimMax, total, true);
    for (int i = 0; i < kGpuStageCount; ++i) {
        const GpuStage stage = static_cast<GpuStage>(i);
        if (isSimulationStage(stage)) {
            stageRow(stageInfo(stage).label, s.gpuAvg[i], s.gpuMax[i], total, false);
        }
    }
    stageRow("Rendering", s.gpuRenderAvg, s.gpuRenderMax, total, true);
    for (int i = 0; i < kGpuStageCount; ++i) {
        const GpuStage stage = static_cast<GpuStage>(i);
        if (!isSimulationStage(stage)) {
            stageRow(stageInfo(stage).label, s.gpuAvg[i], s.gpuMax[i], total, false);
        }
    }
    ImGui::EndTable();
    ImGui::TextDisabled("Per frame, summed over its simulation steps. Share of GPU busy time.");
}

void drawCpuStages(const WindowSummary& s) {
    if (!beginStageTable("cpu_stages")) {
        return;
    }
    for (int i = 0; i < kCpuStageCount; ++i) {
        stageRow(stageInfo(static_cast<CpuStage>(i)).label, s.cpuAvg[i], s.cpuMax[i],
                 s.frameMsAvg, false);
    }
    ImGui::EndTable();
    ImGui::TextDisabled("Wall-clock time on the CPU thread. Share of the frame time.");
}

void drawDiagnostics(FrameProfiler& profiler, const WindowSummary& s) {
    int interval = profiler.diagnosticsInterval();
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::SliderInt("Sample every N frames", &interval, 0, 60, interval == 0 ? "off" : "%d")) {
        profiler.setDiagnosticsInterval(interval);
    }
    if (!s.hasDiagnostics) {
        ImGui::TextDisabled("No samples in this window (paused or sampling off).");
        return;
    }

    const SolverDiagnostics& d = s.latest;
    if (ImGui::BeginTable("diagnostics", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        metricRow("Sim time (s)", "%.3f", d.time);
        metricRow("max |div| before", "%.4g", d.divergenceBeforeMax);
        metricRow("rms div before", "%.4g", d.divergenceBeforeRms);
        metricRow("max |div| after", "%.4g", d.divergenceAfterMax);
        metricRow("rms div after", "%.4g", d.divergenceAfterRms);
        const float reduction = d.divergenceAfterRms > 0.0f
            ? d.divergenceBeforeRms / d.divergenceAfterRms : 0.0f;
        metricRow("rms reduction", "x%.1f", reduction);
        metricRow("Pressure residual", "%.3e", d.pressureResidual, d.pressureResidual > 0.1f);
        metricRow("Max speed", "%.4g", d.maxSpeed);
        metricRow("CFL", "%.3f", d.cfl, d.cfl > 1.0f);
        metricRow("Kinetic energy", "%.4g", d.kineticEnergy);
        metricRow("Smoke mass", "%.4g", d.smokeMass);
        metricRow("Max density", "%.4g", d.maxDensity);
        metricRow("Max temperature", "%.4g", d.maxTemperature);
        ImGui::EndTable();
    }

    char overlay[64];
    std::snprintf(overlay, sizeof(overlay), "log10 max|div| after (%.2f)", safeLog10(d.divergenceAfterMax));
    plotAutoRange("##divergence", s.divergencePlot, overlay, 45.0f);
    std::snprintf(overlay, sizeof(overlay), "log10 residual (%.2f)", safeLog10(d.pressureResidual));
    plotAutoRange("##residual", s.residualPlot, overlay, 45.0f);
    std::snprintf(overlay, sizeof(overlay), "CFL (%.3f)", d.cfl);
    plotAutoRange("##cfl", s.cflPlot, overlay, 45.0f);
}

void setMessage(UiState& ui, bool ok, const std::string& success, const std::string& path) {
    ui.messageIsError = !ok;
    ui.message = ok ? success + path : "Could not write " + path;
}

void drawExport(FrameProfiler& profiler, const ProfilerContext& context, UiState& ui) {
    if (profiler.recording()) {
        if (ImGui::Button("Stop recording")) {
            const uint64_t frames = profiler.recordedFrames();
            const std::string path = profiler.recordingPath();
            profiler.stopRecording();
            setMessage(ui, true, "Recorded " + std::to_string(frames) + " frames to ", path);
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "REC %llu frames",
                           static_cast<unsigned long long>(profiler.recordedFrames()));
    } else if (ImGui::Button("Start recording")) {
        const std::string path = makeProfilePath(context.outputDirectory, "profile_rec");
        const bool ok = profiler.startRecording(path);
        setMessage(ui, ok, "Recording to ", path);
    }

    ImGui::SameLine();
    char label[48];
    std::snprintf(label, sizeof(label), "Export last %s", kWindowLabels[ui.windowChoice]);
    if (ImGui::Button(label)) {
        const std::string path = makeProfilePath(context.outputDirectory, "profile");
        const bool ok = profiler.exportHistory(path, kWindowChoices[ui.windowChoice]);
        setMessage(ui, ok, "Exported to ", path);
    }
    if (!ui.message.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        if (ui.messageIsError) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", ui.message.c_str());
        } else {
            ImGui::TextDisabled("%s", ui.message.c_str());
        }
        ImGui::PopTextWrapPos();
    }
    ImGui::TextDisabled("CSV; summarize with: python src/log_profile.py <file.csv>");
}

void drawInfo(const ProfilerContext& context, UiState& ui) {
    const double now = ImGui::GetTime();
    if (ui.gpuName.empty()) {
        int device = 0;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) == cudaSuccess
            && cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
            ui.gpuName = properties.name;
        } else {
            ui.gpuName = "unknown";
        }
    }
    if (ui.lastMemoryQuery < 0.0 || now - ui.lastMemoryQuery > 0.5) {
        cudaMemGetInfo(&ui.memoryFree, &ui.memoryTotal);
        ui.lastMemoryQuery = now;
    }
    const double mib = 1024.0 * 1024.0;
    ImGui::Text("GPU: %s", ui.gpuName.c_str());
    ImGui::Text("VRAM: %.0f / %.0f MiB used",
                (ui.memoryTotal - ui.memoryFree) / mib, ui.memoryTotal / mib);
    ImGui::Text("Scene: %s%s", context.sceneName, context.paused ? " (paused)" : "");
    ImGui::Text("Grid: %d x %d x %d (%.2f M cells)", context.nx, context.ny, context.nz,
                static_cast<double>(context.nx) * context.ny * context.nz / 1.0e6);
    ImGui::Text("Multigrid levels %d, V-cycles/step %d", context.multigridLevels, context.pressureCycles);
    ImGui::Text("dt %.4f s", context.simulationDt);
}

} // namespace

std::string makeProfilePath(const std::string& directory, const char* prefix) {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &local);
    return directory + prefix + "_" + stamp + ".csv";
}

void drawProfilerWindow(FrameProfiler& profiler, const ProfilerContext& context) {
    UiState& ui = uiState();
    const auto& history = profiler.history();
    const uint64_t newestFrame = history.empty() ? ~uint64_t(0) : history.back().frame;
    const double now = ImGui::GetTime();
    if (newestFrame != ui.lastFrame
        && (ui.lastRefresh < 0.0 || now - ui.lastRefresh >= kRefreshSeconds)) {
        ui.summary = summarize(history, kWindowChoices[ui.windowChoice], context);
        ui.lastRefresh = now;
        ui.lastFrame = newestFrame;
    }

    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420.0f, 740.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.88f);
    bool open = true;
    if (ImGui::Begin("Profiler (F1)", &open)) {
        ImGui::SetNextItemWidth(80.0f);
        if (ImGui::Combo("Window", &ui.windowChoice, kWindowLabels, IM_ARRAYSIZE(kWindowLabels))) {
            ui.lastRefresh = -1.0;
            ui.lastFrame = ~uint64_t(0);
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            profiler.clearHistory();
            ui.summary = WindowSummary{};
        }

        const WindowSummary& s = ui.summary;
        drawFrameRate(s);
        if (ImGui::CollapsingHeader("GPU stages", ImGuiTreeNodeFlags_DefaultOpen)) {
            drawGpuStages(s);
        }
        if (ImGui::CollapsingHeader("CPU stages")) {
            drawCpuStages(s);
        }
        if (ImGui::CollapsingHeader("Solver diagnostics", ImGuiTreeNodeFlags_DefaultOpen)) {
            drawDiagnostics(profiler, s);
        }
        if (ImGui::CollapsingHeader("Export", ImGuiTreeNodeFlags_DefaultOpen)) {
            drawExport(profiler, context, ui);
        }
        if (ImGui::CollapsingHeader("Info")) {
            drawInfo(context, ui);
        }
    }
    ImGui::End();
    if (!open) {
        profiler.setWindowVisible(false);
    }
}
