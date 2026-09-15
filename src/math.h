#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <math.h>

inline float normalize(float value, float minValue, float maxValue) {
    if (maxValue <= minValue)
        return 0.0f;

    return std::clamp(
        (value - minValue) / (maxValue - minValue),
        0.0f,
        1.0f
    );
}

// Convert a float to 8-bit color
inline uint8_t toByte(float x) {
    return static_cast<uint8_t>(
        std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)
    );
}

// Convert pixel coordinate to simulation coordinate
inline float pix2sim(int pix, int display_res, int sim_res) {
    return (pix + 0.5f) / display_res * sim_res - 0.5f;
}

float sampleBilinear(const Grid2D& grid, float gx, float gy) {
    gx = std::clamp(gx, 0.0f, static_cast<float>(grid.width() - 1));
    gy = std::clamp(gy, 0.0f, static_cast<float>(grid.height() - 1));

    const int i0 = static_cast<int>(std::floor(gx));
    const int j0 = static_cast<int>(std::floor(gy));
    const int i1 = std::min(i0 + 1, grid.width() - 1);
    const int j1 = std::min(j0 + 1, grid.height() - 1);

    const float tx = gx - static_cast<float>(i0);
    const float ty = gy - static_cast<float>(j0);
    const float a = std::lerp(grid(i0, j0), grid(i1, j0), tx);
    const float b = std::lerp(grid(i0, j1), grid(i1, j1), tx);
    return std::lerp(a, b, ty);
}