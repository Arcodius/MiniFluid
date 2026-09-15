#include "grid.h"
#include "math.h"

class FluidSolver2D
{
private:
    int nx_;
    int ny_;
    float dx_;

    Grid2D density_next_;

public:
    Grid2D density;
    Grid2D pressure;
    Grid2D divergence;

    Grid2D u;
    Grid2D v;

    float sampleBilinear(const Grid2D& grid, float x, float y) const {
        float gx = x / dx_ - 0.5f;
        float gy = y / dx_ - 0.5f;

        gx = std::clamp(gx, 0.0f, static_cast<float>(nx_ - 1));
        gy = std::clamp(gy, 0.0f, static_cast<float>(ny_ - 1));

        const int i0 = static_cast<int>(std::floor(gx));
        const int j0 = static_cast<int>(std::floor(gy));

        const int i1 = std::min(i0 + 1, nx_ - 1);
        const int j1 = std::min(j0 + 1, ny_ - 1);

        const float tx = gx - static_cast<float>(i0);
        const float ty = gy - static_cast<float>(j0);

        const float q00 = grid(i0, j0);
        const float q10 = grid(i1, j0);
        const float q01 = grid(i0, j1);
        const float q11 = grid(i1, j1);

        const float q0 = (1.0f - tx) * q00 + tx * q10;
        const float q1 = (1.0f - tx) * q01 + tx * q11;

        return (1.0f - ty) * q0 + ty * q1;
    }

    FluidSolver2D(int nx, int ny, float dx) 
        : nx_(nx), ny_(ny), dx_(dx),
        density(nx, ny, dx), pressure(nx, ny, dx), divergence(nx, ny, dx),
        u(nx, ny, dx), v(nx, ny, dx),
        density_next_(nx, ny, dx)
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
                float x = (i+0.5f) * dx_;
                float y = (j+0.5f) * dx_;

                const float rx = x - cx;
                const float ry = y - cy;
                const float r2 = rx * rx + ry * ry;

                // density(i, j) = exp(- r2 / (2.0f * sigma * sigma)); // gaussian blob
                density(i, j) = (0.6 < x && x < 0.8 && 0.1 < y && y < 0.8); // rectangle
            }
        }
    }

    void initVelocity() {
        constexpr float cx = 0.5f;
        constexpr float cy = 0.5f;
        constexpr float omega = 1.0f;

        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const float x = (i+0.5f) * dx_;
                const float y = (j+0.5f) * dx_;

                u(i, j) = -omega * (y - cy);
                v(i, j) =  omega * (x - cx); // rotating field
            }
        }
    }

    std::pair<int, int> getSize() const {
        return std::pair<int, int>(nx_, ny_);
    }

    void advectDensity(float dt) {
        for (int i = 0; i < nx_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                const float x = (i + 0.5f) * dx_;
                const float y = (j + 0.5f) * dx_;

                const float dep_x = x - dt * u(i, j);
                const float dep_y = y - dt * v(i, j);

                density_next_(i, j) = sampleBilinear(density, dep_x, dep_y);
            }
        }
        std::swap(density.data, density_next_.data);
    }

    void step(float dt) {
        advectDensity(dt);
    }
};
