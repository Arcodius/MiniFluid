#include "divergence_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

__global__ void computeDivergenceKernel(
    const float* u,
    const float* v,
    const float* w,
    float* divergence,
    int nx,
    int ny,
    int nz,
    float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    const int cell_index = i + nx * (j + ny * k);
    const int u_stride_x = nx + 1;
    const int u_index = i + u_stride_x * (j + ny * k);
    const int v_index = i + nx * (j + (ny + 1) * k);
    const int w_index = i + nx * (j + ny * k);

    const float du = u[u_index + 1] - u[u_index];
    const float dv = v[v_index + nx] - v[v_index];
    const float dw = w[w_index + nx * ny] - w[w_index];
    divergence[cell_index] = (du + dv + dw) * inv_h;
}

void freeBuffers(float*& u, float*& v, float*& w, float*& divergence) {
    if (u != nullptr) cudaFree(u);
    if (v != nullptr) cudaFree(v);
    if (w != nullptr) cudaFree(w);
    if (divergence != nullptr) cudaFree(divergence);
    u = nullptr;
    v = nullptr;
    w = nullptr;
    divergence = nullptr;
}

} // namespace

CudaDivergence3D::CudaDivergence3D(int nx, int ny, int nz)
    : nx_(nx), ny_(ny), nz_(nz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }

    try {
        checkCuda(cudaMalloc(&device_u_,
            static_cast<size_t>(nx + 1) * ny * nz * sizeof(float)), "allocate u");
        checkCuda(cudaMalloc(&device_v_,
            static_cast<size_t>(nx) * (ny + 1) * nz * sizeof(float)), "allocate v");
        checkCuda(cudaMalloc(&device_w_,
            static_cast<size_t>(nx) * ny * (nz + 1) * sizeof(float)), "allocate w");
        checkCuda(cudaMalloc(&device_divergence_,
            static_cast<size_t>(nx) * ny * nz * sizeof(float)), "allocate divergence");
    } catch (...) {
        freeBuffers(device_u_, device_v_, device_w_, device_divergence_);
        throw;
    }
}

CudaDivergence3D::~CudaDivergence3D() {
    freeBuffers(device_u_, device_v_, device_w_, device_divergence_);
}

void CudaDivergence3D::compute(const MacGrid3D& macgrid, Grid3D& divergence) {
    if (macgrid.nx() != nx_ || macgrid.ny() != ny_ || macgrid.nz() != nz_
        || divergence.width() != nx_ || divergence.height() != ny_
        || divergence.depth() != nz_) {
        throw std::invalid_argument("Grid dimensions do not match CUDA solver");
    }

    checkCuda(cudaMemcpy(device_u_, macgrid.u().data().data(),
        static_cast<size_t>(nx_ + 1) * ny_ * nz_ * sizeof(float),
        cudaMemcpyHostToDevice), "copy u to device");
    checkCuda(cudaMemcpy(device_v_, macgrid.v().data().data(),
        static_cast<size_t>(nx_) * (ny_ + 1) * nz_ * sizeof(float),
        cudaMemcpyHostToDevice), "copy v to device");
    checkCuda(cudaMemcpy(device_w_, macgrid.w().data().data(),
        static_cast<size_t>(nx_) * ny_ * (nz_ + 1) * sizeof(float),
        cudaMemcpyHostToDevice), "copy w to device");

    const dim3 threads(8, 8, 4);
    const dim3 blocks(
        (nx_ + threads.x - 1) / threads.x,
        (ny_ + threads.y - 1) / threads.y,
        (nz_ + threads.z - 1) / threads.z);
    computeDivergenceKernel<<<blocks, threads>>>(
        device_u_, device_v_, device_w_, device_divergence_,
        nx_, ny_, nz_, 1.0f / macgrid.spacing());
    checkCuda(cudaGetLastError(), "launch divergence kernel");

    checkCuda(cudaMemcpy(divergence.data().data(), device_divergence_,
        static_cast<size_t>(nx_) * ny_ * nz_ * sizeof(float),
        cudaMemcpyDeviceToHost), "copy divergence to host");
}