#include "divergence_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

int main() {
    constexpr int nx = 17;
    constexpr int ny = 11;
    constexpr int nz = 7;
    constexpr float h = 0.25f;

    MacGrid3D macgrid(nx, ny, nz, h, 1.0f);
    Grid3D cpu_divergence(nx, ny, nz, h);
    Grid3D cuda_divergence(nx, ny, nz, h);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                macgrid.u()(i, j, k) = 0.1f * i - 0.03f * j + 0.02f * k;
            }
        }
    }
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                macgrid.v()(i, j, k) = -0.04f * i + 0.07f * j + 0.01f * k;
            }
        }
    }
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                macgrid.w()(i, j, k) = 0.02f * i + 0.01f * j - 0.05f * k;
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                cpu_divergence(i, j, k) = macgrid.divergenceAt(i, j, k);
            }
        }
    }

    try {
        CudaDivergence3D cuda_solver(nx, ny, nz);
        cuda_solver.compute(macgrid, cuda_divergence);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA divergence check failed: %s\n", error.what());
        return 1;
    }

    float max_error = 0.0f;
    for (size_t index = 0; index < cpu_divergence.data().size(); ++index) {
        max_error = std::max(max_error, std::abs(
            cpu_divergence.data()[index] - cuda_divergence.data()[index]));
    }

    constexpr float tolerance = 1.0e-5f;
    std::printf("Maximum CPU/CUDA divergence error: %.9g\n", max_error);
    if (max_error > tolerance) {
        std::fprintf(stderr, "Mismatch exceeds tolerance %.1e\n", tolerance);
        return 1;
    }

    std::puts("CUDA divergence check passed");
    return 0;
}