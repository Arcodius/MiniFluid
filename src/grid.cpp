#include "grid.h"

#include "fluid_math.h"
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

// Sample velocity with real coordinates
Vec2 MacGrid2D::sampleVelocity(float x, float y) const {
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

// Velocity correction: only updates the inner faces
void MacGrid2D::applyPressureGradient(float dt) {
    float inv_rho = 1.0f / rho_;
    float inv_h = 1.0f / h_;
    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            u_(i, j) -= dt * inv_rho * (pressure_(i, j) - pressure_(i - 1, j)) * inv_h;
        }
    }
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            v_(i, j) -= dt * inv_rho * (pressure_(i, j) - pressure_(i, j - 1)) * inv_h;
        }
    }
}

Vec3 MacGrid3D::cellVelocity(int i, int j, int k) {
    const float uc = 0.5f * (u_(i, j, k) + u_(i + 1, j, k));
    const float vc = 0.5f * (v_(i, j, k) + v_(i, j + 1, k));
    const float wc = 0.5f * (w_(i, j, k) + w_(i, j, k + 1));
    return Vec3(uc, vc, wc);
}

// Sample velocity with real coordinates
Vec3 MacGrid3D::sampleVelocity(float x, float y, float z) const {
    const float inv_h = 1.0f / h_;

    const float u = sampleTrilinear(
        u_, x * inv_h, y * inv_h - 0.5f, z * inv_h - 0.5f);
    const float v = sampleTrilinear(
        v_, x * inv_h - 0.5f, y * inv_h, z * inv_h - 0.5f);
    const float w = sampleTrilinear(
        w_, x * inv_h - 0.5f, y * inv_h - 0.5f, z * inv_h);

    return Vec3(u, v, w);
}

// Incompressible condition:
// $$ \gradient \cdot u = 0 $$
float MacGrid3D::divergenceAt(int i, int j, int k) {
    const float du = u_(i + 1, j, k) - u_(i, j, k);
    const float dv = v_(i, j + 1, k) - v_(i, j, k);
    const float dw = w_(i, j, k + 1) - w_(i, j, k);
    return (du + dv + dw) / h_;
}

// Velocity correction: only updates the inner faces
void MacGrid3D::applyPressureGradient(float dt) {
    const float inv_rho = 1.0f / rho_;
    const float inv_h = 1.0f / h_;
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 1; i < nx_; ++i) {
                u_(i, j, k) -= dt * inv_rho
                    * (pressure_(i, j, k) - pressure_(i - 1, j, k)) * inv_h;
            }
        }
    }
    for (int k = 0; k < nz_; ++k) {
        for (int j = 1; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                v_(i, j, k) -= dt * inv_rho
                    * (pressure_(i, j, k) - pressure_(i, j - 1, k)) * inv_h;
            }
        }
    }
    for (int k = 1; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                w_(i, j, k) -= dt * inv_rho
                    * (pressure_(i, j, k) - pressure_(i, j, k - 1)) * inv_h;
            }
        }
    }
}

