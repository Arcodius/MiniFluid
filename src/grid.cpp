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

Vec2 MacGrid2D::cellVelocity(int i, int j) {
    const float uc = 0.5f * (u_(i, j) + u_(i + 1, j));
    const float vc = 0.5f * (v_(i, j) + v_(i, j + 1));
    return {uc, vc};
}

Vec2 MacGrid2D::sampleVelocity(float x, float y) {
    const float inv_h = 1.0f / h_;

    const float u = sampleBilinear(u_, x * inv_h, y * inv_h - 0.5f);
    const float v = sampleBilinear(v_, x * inv_h - 0.5f, y * inv_h);

    return {u, v};
}

// Incompressible condition: 
// $$ \gradient \cdot u = 0 $$
float MacGrid2D::divergenceAt(int i, int j) {
    const float du = u_(i + 1, j) - u_(i, j);
    const float dv = v_(i, j + 1) - v_(i, j);
    return (du + dv) / h_;
}

