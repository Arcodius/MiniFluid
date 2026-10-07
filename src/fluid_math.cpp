#include "fluid_math.h"
#include "grid.h"

// input: grid-index coordinates
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

float sampleTrilinear(const Grid3D& grid, float gx, float gy, float gz) {
    gx = std::clamp(gx, 0.0f, static_cast<float>(grid.width() - 1));
    gy = std::clamp(gy, 0.0f, static_cast<float>(grid.height() - 1));
    gz = std::clamp(gz, 0.0f, static_cast<float>(grid.depth() - 1));

    const int i0 = static_cast<int>(std::floor(gx));
    const int j0 = static_cast<int>(std::floor(gy));
    const int k0 = static_cast<int>(std::floor(gz));
    const int i1 = std::min(i0 + 1, grid.width() - 1);
    const int j1 = std::min(j0 + 1, grid.height() - 1);
    const int k1 = std::min(k0 + 1, grid.depth() - 1);

    const float tx = gx - static_cast<float>(i0);
    const float ty = gy - static_cast<float>(j0);
    const float tz = gz - static_cast<float>(k0);
    const float a = std::lerp(grid(i0, j0, k0), grid(i1, j0, k0), tx);
    const float b = std::lerp(grid(i0, j1, k0), grid(i1, j1, k0), tx);
    const float v1 = std::lerp(a, b, ty);
    const float c = std::lerp(grid(i0, j0, k1), grid(i1, j0, k1), tx);
    const float d = std::lerp(grid(i0, j1, k1), grid(i1, j1, k1), tx);
    const float v2 = std::lerp(c, d, ty);
    return std::lerp(v1, v2, tz);
}