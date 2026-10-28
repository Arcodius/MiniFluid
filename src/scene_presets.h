#pragma once

#include "cuda/cuda_fluid_solver.h"

#include <vector>

struct ScenePreset {
    const char* name;
    SmokeParameters parameters;
    std::vector<SmokeEmitter> emitters;
};

// Built-in smoke scenes for a domain that is 1 world unit wide and deep.
std::vector<ScenePreset> makeScenePresets();
