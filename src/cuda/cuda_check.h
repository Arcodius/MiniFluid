#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <cstddef>
#include <string>

void checkCuda(cudaError_t result, const char* operation, const char* file, int line);
size_t checkedElementCount(int nx, int ny, int nz);
int checkedMacDimension(int dimension);
void reportCudaCleanupError(cudaError_t result) noexcept;
void write_to_log_cuda(const std::string& kernel_name, float ms);
void write_to_log(const std::string& kernel_name, float ms);

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define CUDA_TIME(call)                        \
    do {                                                    \
        cudaStream_t stream = 0;                            \
        cudaEvent_t start, stop;                            \
        CUDA_CHECK(cudaEventCreate(&start));                \
        CUDA_CHECK(cudaEventCreate(&stop));                 \
        CUDA_CHECK(cudaEventRecord(start, stream));         \
        call;                                        \
        CUDA_CHECK(cudaGetLastError());                     \
        CUDA_CHECK(cudaEventRecord(stop, stream));          \
        CUDA_CHECK(cudaEventSynchronize(stop));             \
        float ms = 0.0f;                                    \
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); \
        write_to_log_cuda(#call, ms);                             \
        CUDA_CHECK(cudaEventDestroy(start));                \
        CUDA_CHECK(cudaEventDestroy(stop));                 \
    } while (false)
#define TIME(call) \
    do {  \
        auto start = std::chrono::high_resolution_clock::now();              \
        call;   \
        CUDA_CHECK(cudaStreamSynchronize(0)); \
        auto end = std::chrono::high_resolution_clock::now();\
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);\
        write_to_log(#call, static_cast<float>(duration.count()));\
    } while (false)

#define REPLACE(call) call