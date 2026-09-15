#include "grid.h"

#include "math.h"
#include <cassert>

float& Grid2D::operator() (int i, int j) {
    assert(i >= 0 && i < nx_);
    assert(j >= 0 && j < ny_);
    return data_[i + nx_ * j];
}

float Grid2D::operator() (int i, int j) const {
    assert(i >= 0 && i < nx_);
    assert(j >= 0 && j < ny_);
    return data_[i + nx_ * j];
}

float MacGrid2D::bilerpClamped(const Grid2D& grid, float gx, float gy) {
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

Vec2 MacGrid2D::cellVelocity(int i, int j) {
    const float uc = 0.5f * (u_(i, j) + u_(i + 1, j));
    const float vc = 0.5f * (v_(i, j) + v_(i, j + 1));
    return {uc, vc};
}

Vec2 MacGrid2D::sampleVelocity(float x, float y) {
    const float inv_h = 1.0f / h_;

    const float u = bilerpClamped(u_, x * inv_h, y * inv_h - 0.5f);
    const float v = bilerpClamped(v_, x * inv_h - 0.5f, y * inv_h);

    return {u, v};
}

// Incompressible condition: 
// $$ \gradient \cdot u = 0 $$
float MacGrid2D::divergenceAt(int i, int j) {
    const float du = u_(i + 1, j) - u_(i, j);
    const float dv = v_(i, j + 1) - v_(i, j);
    return (du + dv) / h_;
}

