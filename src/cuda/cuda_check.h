#pragma once

#include <cuda_runtime.h>

#include <cstddef>

void checkCuda(cudaError_t result, const char* operation, const char* file, int line);
size_t checkedElementCount(int nx, int ny, int nz);
int checkedMacDimension(int dimension);
void reportCudaCleanupError(cudaError_t result) noexcept;

#define CHECK_CUDA(call) checkCuda((call), #call, __FILE__, __LINE__)
