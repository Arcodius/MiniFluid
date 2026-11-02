#pragma once

#include <string>

class FrameProfiler;

// Scene and solver facts shown in the profiler window. Owned by the application.
struct ProfilerContext {
    const char* sceneName = "";
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int multigridLevels = 0;
    int pressureCycles = 0;
    double simulationDt = 0.0;
    bool paused = false;
    std::string outputDirectory;  // Exported CSV files go here; ends with a separator or is empty
};

// Draws the profiler window inside the current ImGui frame. Closing the window hides it.
void drawProfilerWindow(FrameProfiler& profiler, const ProfilerContext& context);

// <directory><prefix>_YYYYmmdd_HHMMSS.csv
std::string makeProfilePath(const std::string& directory, const char* prefix);
