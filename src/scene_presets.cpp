#include "scene_presets.h"

namespace {

ScenePreset makePlume() {
    ScenePreset preset{"Buoyant plume", SmokeParameters{}, {}};

    SmokeEmitter source;
    source.center = make_float3(0.5f, 0.07f, 0.5f);
    source.radius = 0.075f;
    source.velocity = make_float3(0.0f, 0.3f, 0.0f);
    source.density = 1.0f;
    source.temperature = 1.0f;
    source.noiseAmplitude = 0.7f;
    source.velocityJitter = 0.35f;
    source.noiseFrequency = 9.0f;
    preset.emitters.push_back(source);
    return preset;
}

ScenePreset makeCollision() {
    ScenePreset preset{"Colliding jets", SmokeParameters{}, {}};
    preset.parameters.buoyancyTemperature = 1.8f;
    preset.parameters.vorticity = 3.5f;
    preset.parameters.densityDissipation = 0.12f;

    SmokeEmitter left;
    left.center = make_float3(0.12f, 0.2f, 0.47f);
    left.radius = 0.045f;
    left.velocity = make_float3(1.0f, 0.1f, 0.0f);
    left.temperature = 0.8f;
    left.noiseAmplitude = 0.5f;
    left.velocityJitter = 0.3f;
    preset.emitters.push_back(left);

    SmokeEmitter right = left;
    right.center = make_float3(0.88f, 0.2f, 0.53f);
    right.velocity = make_float3(-1.0f, 0.1f, 0.0f);
    preset.emitters.push_back(right);
    return preset;
}

// Short pulses from a thin disk roll up into vortex rings.
ScenePreset makeRings() {
    ScenePreset preset{"Smoke rings", SmokeParameters{}, {}};
    preset.parameters.buoyancyTemperature = 0.4f;
    preset.parameters.vorticity = 0.8f;
    preset.parameters.densityDissipation = 0.02f;

    SmokeEmitter puffer;
    puffer.center = make_float3(0.5f, 0.06f, 0.5f);
    puffer.radius = 0.11f;
    puffer.thickness = 0.025f;
    puffer.velocity = make_float3(0.0f, 2.6f, 0.0f);
    puffer.temperature = 0.3f;
    puffer.noiseAmplitude = 0.2f;
    puffer.velocityJitter = 0.05f;
    puffer.pulsePeriod = 1.4f;
    puffer.pulseDuty = 0.06f;
    preset.emitters.push_back(puffer);
    return preset;
}

ScenePreset makeSwirl() {
    ScenePreset preset{"Wandering source", SmokeParameters{}, {}};
    preset.parameters.vorticity = 4.0f;

    SmokeEmitter source;
    source.center = make_float3(0.5f, 0.07f, 0.5f);
    source.radius = 0.06f;
    source.velocity = make_float3(0.0f, 0.4f, 0.0f);
    source.temperature = 1.1f;
    source.noiseAmplitude = 0.7f;
    source.velocityJitter = 0.4f;
    source.orbitRadius = 0.2f;
    source.orbitSpeed = 1.3f;
    preset.emitters.push_back(source);
    return preset;
}

}

std::vector<ScenePreset> makeScenePresets() {
    return {makePlume(), makeCollision(), makeRings(), makeSwirl()};
}
