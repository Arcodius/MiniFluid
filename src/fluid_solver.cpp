#include "fluid_solver.h"

void FluidSolver2D::initDensity() {
    // // Simple gaussian blob
    // constexpr float cx = 0.75f;
    // constexpr float cy = 0.50f;
    // constexpr float sigma = 0.06f;

    // for (int j = 0; j < ny_; ++j) {
    //     for (int i = 0; i < nx_; ++i) {
    //         const float x = (i + 0.5f) * h_;
    //         const float y = (j + 0.5f) * h_;

    //         const float rx = x - cx;
    //         const float ry = y - cy;
    //         const float r2 = rx * rx + ry * ry;

    //         density(i, j) = exp(- r2 / (2.0f * sigma * sigma)); // gaussian blob
    //         // density(i, j) = (0.6f < x && x < 0.8f && 0.1f < y && y < 0.8f); // rectangle
    //     }
    // }
}

void FluidSolver2D::initVelocity() {
    // constexpr float cx = 0.5f;
    // constexpr float cy = 0.5f;
    // constexpr float omega = 1.0f;

    // for (int j = 0; j < ny_; ++j) {
    //     for (int i = 0; i <= nx_; ++i) {
    //         const float x = i * h_;
    //         const float y = j * h_;

    //         macgrid.u()(i, j) = -omega * (y - cy);
    //     }
    // }

    // for (int j = 0; j <= ny_; ++j) {
    //     for (int i = 0; i < nx_; ++i) {
    //         const float x = i * h_;
    //         const float y = j * h_;

    //         macgrid.v()(i, j) = omega * (x - cx); // rotating field
    //     }
    // }
    
    macgrid.v().fill(0.0f);
    macgrid.u().fill(0.0f);
}

std::pair<int, int> FluidSolver2D::getSize() const {
    return std::pair<int, int>(nx_, ny_);
}

void FluidSolver2D::advectDensity(float dt) {
    const float inv_h = 1.0f / h_;
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            const float x = (i + 0.5f) * h_;
            const float y = (j + 0.5f) * h_;

            const Vec2 vel = macgrid.sampleVelocity(x, y);
            const float dep_x = x - dt * vel.x;
            const float dep_y = y - dt * vel.y;

            const float dep_i = dep_x * inv_h - 0.5f;
            const float dep_j = dep_y * inv_h - 0.5f;
            density_next_(i, j) = sampleBilinear(density, dep_i, dep_j);
        }
    }
    std::swap(density.data(), density_next_.data());
}

void FluidSolver2D::computeDivergence() {
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            divergence(i, j) = macgrid.divergenceAt(i, j);
        }
    }
}

void FluidSolver2D::solvePressure(float dt, int iterations) {
    macgrid.pressure().fill(0.0f);
    const float scale = h_ * h_ * rho_ / dt;
    Grid2D pressure_new = Grid2D(nx_, ny_, h_);
    for (int iter = 0; iter < iterations; ++iter) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                float left = macgrid.pressure()(std::clamp(i - 1, 0, nx_ - 1), j);
                float right = macgrid.pressure()(std::clamp(i + 1, 0, nx_ - 1), j);
                float down = macgrid.pressure()(i, std::clamp(j - 1, 0, ny_ - 1));
                float up = macgrid.pressure()(i, std::clamp(j + 1, 0, ny_ - 1));
                pressure_new(i, j) = (left + right + down + up - scale * divergence(i, j)) / 4.0;
            }
        }
        std::swap(pressure_new, macgrid.pressure());
    }
}

void FluidSolver2D::applyPressureGradient(float dt) {
    macgrid.applyPressureGradient(dt);
}

void FluidSolver2D::testDivergence() {
    float max_div = 0.0f;
    float sum_sqr = 0.0f;

    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            float div = divergence(i, j);
            max_div = std::max(max_div, std::abs(div));
            sum_sqr += div * div;
        }
    }

    float rms_div = std::sqrt(sum_sqr / (nx_ * ny_));
    printf("Divergence max-%f, rms-%f\n", max_div, rms_div);
}

void FluidSolver2D::step(float dt) {
    advectDensity(dt);
    computeDivergence();
    testDivergence();

    solvePressure(dt, 100);
    applyPressureGradient(dt);

    computeDivergence();
    testDivergence();
}