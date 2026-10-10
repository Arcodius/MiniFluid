#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>

void checkCuda(cudaError_t result, const char* operation, const char* file, int line);
size_t checkedElementCount(int nx, int ny, int nz);
int checkedMacDimension(int dimension);
void reportCudaCleanupError(cudaError_t result) noexcept;

void write_to_log(const std::string& kernel_name, float ms);

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define CUDA_TIME(name, stream, ...)                        \
    do {                                                    \
        cudaEvent_t start, stop;                            \
        CUDA_CHECK(cudaEventCreate(&start));                \
        CUDA_CHECK(cudaEventCreate(&stop));                 \
        CUDA_CHECK(cudaEventRecord(start, stream));         \
        __VA_ARGS__;                                        \
        CUDA_CHECK(cudaGetLastError());                     \
        CUDA_CHECK(cudaEventRecord(stop, stream));          \
        CUDA_CHECK(cudaEventSynchronize(stop));             \
        float ms = 0.0f;                                    \
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); \
        write_to_log(name, ms);                             \
        CUDA_CHECK(cudaEventDestroy(start));                \
        CUDA_CHECK(cudaEventDestroy(stop));                 \
    } while (false)