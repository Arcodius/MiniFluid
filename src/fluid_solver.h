#pragma once
#include "grid.h"
#include "math.h"

class FluidSolver2D
{
private:
    int nx_;
    int ny_;
    float h_;

    Grid2D density_next_;

public:
    Grid2D density;
    Grid2D divergence;
    MacGrid2D macgrid; // includes pressure


    FluidSolver2D(int nx, int ny, float h) 
        : nx_(nx), ny_(ny), h_(h),
        density(nx, ny, h), divergence(nx, ny, h),
        macgrid(nx, ny, h),
        density_next_(nx, ny, h)
    {
        initVelocity();
        initDensity();
    }

    void initDensity() {
        // Simple gaussian blob
        constexpr float cx = 0.75f;
        constexpr float cy = 0.50f;
        constexpr float sigma = 0.06f;

        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const float x = (i + 0.5f) * h_;
                const float y = (j + 0.5f) * h_;

                const float rx = x - cx;
                const float ry = y - cy;
                const float r2 = rx * rx + ry * ry;

                density(i, j) = exp(- r2 / (2.0f * sigma * sigma)); // gaussian blob
                // density(i, j) = (0.6f < x && x < 0.8f && 0.1f < y && y < 0.8f); // rectangle
            }
        }
    }

    void initVelocity() {
        constexpr float cx = 0.5f;
        constexpr float cy = 0.5f;
        constexpr float omega = 1.0f;

        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i <= nx_; ++i) {
                const float x = i * h_;
                const float y = j * h_;

                macgrid.u()(i, j) = -omega * (y - cy);
            }
        }

        for (int j = 0; j <= ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const float x = i * h_;
                const float y = j * h_;

                macgrid.v()(i, j) = omega * (x - cx); // rotating field
            }
        }
    }

    std::pair<int, int> getSize() const {
        return std::pair<int, int>(nx_, ny_);
    }

    void advectDensity(float dt) {
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

    void computeDivergence() {
        for (int i = 0; i < nx_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                divergence(i, j) = macgrid.divergenceAt(i, j);
            }
        }
    }

    void step(float dt) {
        advectDensity(dt);
        computeDivergence();
    }
};
