#include "cuda_fluid_solver.h"

#include <algorithm>
#include <cmath>

namespace {

void clearDeviceGrid(DeviceGrid3D& grid) {
    const GridView3D<float> view = grid.view();
    CUDA_CHECK(cudaMemset(view.data, 0, view.size() * sizeof(float)));
}

void copySliceToHost(
    GridView3D<const float> grid,
    int k,
    float* destination,
    size_t element_count) {
    const size_t slice_size = static_cast<size_t>(grid.nx) * grid.ny;
    if (k < 0 || k >= grid.nz) {
        throw std::out_of_range("Slice index is outside the grid");
    }
    if (destination == nullptr || element_count != slice_size) {
        throw std::invalid_argument("Destination size does not match the grid slice");
    }

    const size_t offset = static_cast<size_t>(k) * slice_size;
    CUDA_CHECK(cudaMemcpy(
        destination, grid.data + offset, slice_size * sizeof(float),
        cudaMemcpyDeviceToHost));
}

__device__ float clampCoordinate(float coordinate, int extent) {
    return fminf(fmaxf(coordinate, 0.0f), static_cast<float>(extent - 1));
}

__device__ float sampleTrilinear(
    GridView3D<const float> grid, float x, float y, float z) {
    x = clampCoordinate(x, grid.nx);
    y = clampCoordinate(y, grid.ny);
    z = clampCoordinate(z, grid.nz);

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int z0 = static_cast<int>(floorf(z));
    const int x1 = min(x0 + 1, grid.nx - 1);
    const int y1 = min(y0 + 1, grid.ny - 1);
    const int z1 = min(z0 + 1, grid.nz - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float tz = z - static_cast<float>(z0);

    const float x00 = grid(x0, y0, z0) + tx * (grid(x1, y0, z0) - grid(x0, y0, z0));
    const float x10 = grid(x0, y1, z0) + tx * (grid(x1, y1, z0) - grid(x0, y1, z0));
    const float x01 = grid(x0, y0, z1) + tx * (grid(x1, y0, z1) - grid(x0, y0, z1));
    const float x11 = grid(x0, y1, z1) + tx * (grid(x1, y1, z1) - grid(x0, y1, z1));
    const float y0_value = x00 + ty * (x10 - x00);
    const float y1_value = x01 + ty * (x11 - x01);
    return y0_value + tz * (y1_value - y0_value);
}

__device__ void sampleVelocity(
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float x,
    float y,
    float z,
    float inv_h,
    float& velocity_x,
    float& velocity_y,
    float& velocity_z) {
    velocity_x = sampleTrilinear(u, x * inv_h, y * inv_h - 0.5f, z * inv_h - 0.5f);
    velocity_y = sampleTrilinear(v, x * inv_h - 0.5f, y * inv_h, z * inv_h - 0.5f);
    velocity_z = sampleTrilinear(w, x * inv_h - 0.5f, y * inv_h - 0.5f, z * inv_h);
}

template<int Component>
__global__ void advectVelocityKernel(
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    GridView3D<float> output,
    float h,
    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= output.nx || j >= output.ny || k >= output.nz) {
        return;
    }

    const float x = (static_cast<float>(i) + (Component == 0 ? 0.0f : 0.5f)) * h;
    const float y = (static_cast<float>(j) + (Component == 1 ? 0.0f : 0.5f)) * h;
    const float z = (static_cast<float>(k) + (Component == 2 ? 0.0f : 0.5f)) * h;

    float velocity_x;
    float velocity_y;
    float velocity_z;
    sampleVelocity(u, v, w, x, y, z, 1.0f / h,
        velocity_x, velocity_y, velocity_z);

    const float departure_x = x - dt * velocity_x;
    const float departure_y = y - dt * velocity_y;
    const float departure_z = z - dt * velocity_z;
    const float grid_x = departure_x / h - (Component == 0 ? 0.0f : 0.5f);
    const float grid_y = departure_y / h - (Component == 1 ? 0.0f : 0.5f);
    const float grid_z = departure_z / h - (Component == 2 ? 0.0f : 0.5f);
    output(i, j, k) = sampleTrilinear(
        Component == 0 ? u : (Component == 1 ? v : w),
        grid_x, grid_y, grid_z);
}

template<int Component>
void launchAdvectVelocity(
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    GridView3D<float> output,
    float h,
    float dt) {
    const dim3 threads(8, 8, 4);
    const dim3 blocks(
        (output.nx + threads.x - 1) / threads.x,
        (output.ny + threads.y - 1) / threads.y,
        (output.nz + threads.z - 1) / threads.z);
    advectVelocityKernel<Component><<<blocks, threads>>>(
        u, v, w, output, h, dt);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void addVelocitySourceKernel(
    GridView3D<float> v,
    int nx,
    int nz,
    float v0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) {
        return;
    }

    constexpr float radius = 4.0f;
    const float center_x = 0.5f * static_cast<float>(nx);
    const float center_z = 0.5f * static_cast<float>(nz);
    const float dx = static_cast<float>(i) + 0.5f - center_x;
    const float dz = static_cast<float>(k) + 0.5f - center_z;
    const float distance_squared = dx * dx + dz * dz;
    const float weight = expf(-distance_squared / (2.0f * radius * radius));

    if (weight > 0.01f) {
        v(i, 1, k) = v0 * weight;
    }
}

__global__ void addDensitySourceKernel(
    GridView3D<float> density,
    int begin_x,
    int begin_z,
    int patch_nx,
    int patch_nz,
    float rho) {
    const int patch_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int patch_k = blockIdx.y * blockDim.y + threadIdx.y;
    if (patch_i >= patch_nx || patch_k >= patch_nz) {
        return;
    }

    const int i = begin_x + patch_i;
    const int k = begin_z + patch_k;
    density(i, 0, k) = rho;
    if (density.ny > 1) {
        density(i, 1, k) = rho;
    }
}

__global__ void zeroUBoundaryKernel(GridView3D<float> u, int nx) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < u.ny && k < u.nz) {
        u(0, j, k) = 0.0f;
        u(nx, j, k) = 0.0f;
    }
}

__global__ void zeroVBoundaryKernel(GridView3D<float> v, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < v.nx && k < v.nz) {
        v(i, 0, k) = 0.0f;
        v(i, ny, k) = 0.0f;
    }
}

__global__ void zeroWBoundaryKernel(GridView3D<float> w, int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < w.nx && j < w.ny) {
        w(i, j, 0) = 0.0f;
        w(i, j, nz) = 0.0f;
    }
}

__global__ void computeDivergenceKernel(
    GridView3D<float> d,
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= d.nx || j >= d.ny || k >= d.nz) return;

    const float du = u(i + 1, j, k) - u(i, j, k);
    const float dv = v(i, j + 1, k) - v(i, j, k);
    const float dw = w(i, j, k + 1) - w(i, j, k);
    d(i, j, k) = (du + dv + dw) / h;
}

__global__ void solvePressureKernel(
    GridView3D<const float> pressure,
    GridView3D<const float> divergence,
    GridView3D<float> pressure_next,
    float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= pressure.nx || j >= pressure.ny || k >= pressure.nz) {
        return;
    }

    const int i_left = max(i - 1, 0);
    const int i_right = min(i + 1, pressure.nx - 1);
    const int j_down = max(j - 1, 0);
    const int j_up = min(j + 1, pressure.ny - 1);
    const int k_back = max(k - 1, 0);
    const int k_front = min(k + 1, pressure.nz - 1);

    pressure_next(i, j, k) = (
        pressure(i_left, j, k) + pressure(i_right, j, k)
        + pressure(i, j_down, k) + pressure(i, j_up, k)
        + pressure(i, j, k_back) + pressure(i, j, k_front)
        - scale * divergence(i, j, k)) / 6.0f;
}

template<int Component>
__global__ void pressureGradientKernel(
    GridView3D<float> velocity,
    GridView3D<const float> pressure,
    float inv_rho,
    float inv_h,
    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= velocity.nx || j >= velocity.ny || k >= velocity.nz) {
        return;
    }

    float pressure_difference;
    if constexpr (Component == 0) {
        if (i == 0 || i >= pressure.nx) return;
        pressure_difference = pressure(i, j, k) - pressure(i - 1, j, k);
    } else if constexpr (Component == 1) {
        if (j == 0 || j >= pressure.ny) return;
        pressure_difference = pressure(i, j, k) - pressure(i, j - 1, k);
    } else {
        if (k == 0 || k >= pressure.nz) return;
        pressure_difference = pressure(i, j, k) - pressure(i, j, k - 1);
    }

    velocity(i, j, k) -= dt * inv_rho * pressure_difference * inv_h;
}

template<int Component>
void launchPressureGradientKernel(
    GridView3D<float> velocity,
    GridView3D<const float> pressure,
    float inv_rho, float inv_h, float dt) {
    const dim3 threads(8, 8, 4);
    const dim3 blocks(
        (velocity.nx + threads.x - 1) / threads.x,
        (velocity.ny + threads.y - 1) / threads.y,
        (velocity.nz + threads.z - 1) / threads.z);
    pressureGradientKernel<Component><<<blocks, threads>>>(
        velocity, pressure, inv_rho, inv_h, dt);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void advectDensityKernel(
    GridView3D<const float> density, GridView3D<float> density_next, 
    GridView3D<const float> u, GridView3D<const float> v, GridView3D<const float> w,
    float dt, float h, float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= density.nx || j >= density.ny || k >= density.nz) {
        return;
    }

    const float x = (i + 0.5f) * h;
    const float y = (j + 0.5f) * h;
    const float z = (k + 0.5f) * h;

    float velocity_x;
    float velocity_y;
    float velocity_z;
    sampleVelocity(u, v, w, x, y, z, inv_h, velocity_x, velocity_y, velocity_z);
    const float dep_x = x - dt * velocity_x;
    const float dep_y = y - dt * velocity_y;
    const float dep_z = z - dt * velocity_z;

    const float dep_i = dep_x * inv_h - 0.5f;
    const float dep_j = dep_y * inv_h - 0.5f;
    const float dep_k = dep_z * inv_h - 0.5f;
    density_next(i, j, k) = sampleTrilinear(density, dep_i, dep_j, dep_k);
}

} // namespace

CudaFluidSolver::CudaFluidSolver(int nx, int ny, int nz, float h, float rho)
    : macgrid_(nx, ny, nz, h, rho),
      u_next_(nx + 1, ny, nz),
      v_next_(nx, ny + 1, nz),
      w_next_(nx, ny, nz + 1),
      density_(nx, ny, nz),
      divergence_(nx, ny, nz),
      density_next_(nx, ny, nz),
      pressure_next_(nx, ny, nz) {
    initDensity();
    initVelocity();
    clearDeviceGrid(divergence_);
}

CudaFluidSolver::~CudaFluidSolver() = default;

void CudaFluidSolver::copyDensitySlice(
    int k, float* destination, size_t element_count) const {
    copySliceToHost(
        static_cast<const DeviceGrid3D&>(density_).view(),
        k, destination, element_count);
}

void CudaFluidSolver::copyDivergenceSlice(
    int k, float* destination, size_t element_count) const {
    copySliceToHost(
        static_cast<const DeviceGrid3D&>(divergence_).view(),
        k, destination, element_count);
}

void CudaFluidSolver::initDensity() {
    clearDeviceGrid(density_);
}
void CudaFluidSolver::initVelocity() {
    clearDeviceGrid(macgrid_.u());
    clearDeviceGrid(macgrid_.v());
    clearDeviceGrid(macgrid_.w());
}


void CudaFluidSolver::advectVelocity(float dt) {
    const float h = macgrid_.spacing();
    const auto& host_view = static_cast<const DeviceMacGrid3D&>(macgrid_);
    const auto u = host_view.u().view();
    const auto v = host_view.v().view();
    const auto w = host_view.w().view();

    launchAdvectVelocity<0>(u, v, w, u_next_.view(), h, dt);
    launchAdvectVelocity<1>(u, v, w, v_next_.view(), h, dt);
    launchAdvectVelocity<2>(u, v, w, w_next_.view(), h, dt);

    std::swap(macgrid_.u(), u_next_);
    std::swap(macgrid_.v(), v_next_);
    std::swap(macgrid_.w(), w_next_);
}

void CudaFluidSolver::addVelocitySource(float v0) {
    const dim3 threads(16, 16);
    const dim3 blocks(
        (macgrid_.nx() + threads.x - 1) / threads.x,
        (macgrid_.nz() + threads.y - 1) / threads.y);
    addVelocitySourceKernel<<<blocks, threads>>>(
        macgrid_.v().view(), macgrid_.nx(), macgrid_.nz(), v0);
    CUDA_CHECK(cudaGetLastError());
}

void CudaFluidSolver::addDensitySource(float rho) {
    constexpr int half_width = 3;
    const int center_x = (macgrid_.nx() + 1) / 2;
    const int center_z = (macgrid_.nz() + 1) / 2;
    const int begin_x = std::max(center_x - half_width, 0);
    const int begin_z = std::max(center_z - half_width, 0);
    const int end_x = std::min(center_x + half_width, density_.nx() - 1);
    const int end_z = std::min(center_z + half_width, density_.nz() - 1);
    if (begin_x > end_x || begin_z > end_z) {
        return;
    }

    const int patch_nx = end_x - begin_x + 1;
    const int patch_nz = end_z - begin_z + 1;
    const dim3 threads(16, 16);
    const dim3 blocks(
        (patch_nx + threads.x - 1) / threads.x,
        (patch_nz + threads.y - 1) / threads.y);
    addDensitySourceKernel<<<blocks, threads>>>(
        density_.view(), begin_x, begin_z, patch_nx, patch_nz, rho);
    CUDA_CHECK(cudaGetLastError());
}

void CudaFluidSolver::addForces(float dt) {

}

void CudaFluidSolver::enforceBoundaryVelocity() {
    constexpr int tile = 16;
    const dim3 threads(tile, tile);

    const GridView3D<float> u = macgrid_.u().view();
    const dim3 u_blocks(
        (u.ny + tile - 1) / tile,
        (u.nz + tile - 1) / tile);
    zeroUBoundaryKernel<<<u_blocks, threads>>>(u, macgrid_.nx());
    CUDA_CHECK(cudaGetLastError());

    const GridView3D<float> v = macgrid_.v().view();
    const dim3 v_blocks(
        (v.nx + tile - 1) / tile,
        (v.nz + tile - 1) / tile);
    zeroVBoundaryKernel<<<v_blocks, threads>>>(v, macgrid_.ny());
    CUDA_CHECK(cudaGetLastError());

    const GridView3D<float> w = macgrid_.w().view();
    const dim3 w_blocks(
        (w.nx + tile - 1) / tile,
        (w.ny + tile - 1) / tile);
    zeroWBoundaryKernel<<<w_blocks, threads>>>(w, macgrid_.nz());
    CUDA_CHECK(cudaGetLastError());
}

void CudaFluidSolver::computeDivergence() {
    const dim3 threads(8, 8, 4);
    const GridView3D<float> d = divergence_.view();
    const dim3 blocks(
        (d.nx + threads.x - 1) / threads.x,
        (d.ny + threads.y - 1) / threads.y,
        (d.nz + threads.z - 1) / threads.z);

    const auto& macgrid = static_cast<const DeviceMacGrid3D&>(macgrid_);
    computeDivergenceKernel<<<blocks, threads>>>(
        d,
        macgrid.u().view(),
        macgrid.v().view(),
        macgrid.w().view(),
        macgrid.spacing());
    CUDA_CHECK(cudaGetLastError());
}

void CudaFluidSolver::solvePressure(float dt, int iterations) {
    const GridView3D<float> initial_pressure = macgrid_.pressure().view();
    CUDA_CHECK(cudaMemset(initial_pressure.data, 0, initial_pressure.size() * sizeof(float)));

    const float scale = macgrid_.spacing() * macgrid_.spacing() * macgrid_.rho() / dt;
    const dim3 threads(8, 8, 4);
    const dim3 blocks(
        (initial_pressure.nx + threads.x - 1) / threads.x,
        (initial_pressure.ny + threads.y - 1) / threads.y,
        (initial_pressure.nz + threads.z - 1) / threads.z);
    const auto divergence = static_cast<const DeviceGrid3D&>(divergence_).view();

    for (int iter = 0; iter < iterations; ++iter) {
        const auto pressure = static_cast<const DeviceGrid3D&>(macgrid_.pressure()).view();
        solvePressureKernel<<<blocks, threads>>>(pressure, divergence, pressure_next_.view(), scale);
        CUDA_CHECK(cudaGetLastError());
        std::swap(macgrid_.pressure(), pressure_next_);
    }
}

void CudaFluidSolver::applyPressureGradient(float dt) {
    const float inv_rho = 1.0f / macgrid_.rho();
    const float inv_h = 1.0f / macgrid_.spacing();
    const auto& macgrid = static_cast<const DeviceMacGrid3D&>(macgrid_);
    const auto pressure = macgrid.pressure().view();

    launchPressureGradientKernel<0>(
        macgrid_.u().view(), pressure, inv_rho, inv_h, dt);
    launchPressureGradientKernel<1>(
        macgrid_.v().view(), pressure, inv_rho, inv_h, dt);
    launchPressureGradientKernel<2>(
        macgrid_.w().view(), pressure, inv_rho, inv_h, dt);
}

void CudaFluidSolver::advectDensity(float dt) {
    const float h = macgrid_.spacing();
    const float inv_h = 1.0f / h;
    const dim3 threads(8, 8, 4);
    const DeviceGrid3D& density = static_cast<const DeviceGrid3D&>(density_);
    const dim3 blocks(
        (density_.nx() + threads.x - 1) / threads.x,
        (density_.ny() + threads.y - 1) / threads.y,
        (density_.nz() + threads.z - 1) / threads.z);
    const auto& host_view = static_cast<const DeviceMacGrid3D&>(macgrid_);
    const auto u = host_view.u().view();
    const auto v = host_view.v().view();
    const auto w = host_view.w().view();
    advectDensityKernel<<<blocks, threads>>>(
        density.view(), density_next_.view(), u, v, w, dt, h, inv_h);
    CUDA_CHECK(cudaGetLastError());
    std::swap(density_, density_next_);
}

void CudaFluidSolver::step(float dt) {
    TIME(advectVelocity(dt));
    TIME(addVelocitySource(1.0f));
    TIME(addDensitySource(1.0f));
    TIME(addForces(dt));
    TIME(enforceBoundaryVelocity());

    TIME(computeDivergence());
    // testDivergence();
    TIME(solvePressure(dt, 100));
    TIME(applyPressureGradient(dt));
    // computeDivergence();
    // testDivergence();

    TIME(advectDensity(dt));
}