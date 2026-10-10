#include "cuda_check.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

void checkCuda(cudaError_t result, const char* operation, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            "CUDA Error at " + std::string(file) + ":" + std::to_string(line)
            + "\n  Operation: " + operation
            + "\n  Error Code: " + std::to_string(result)
            + "\n  Error String: " + cudaGetErrorString(result));
    }
}

size_t checkedElementCount(int nx, int ny, int nz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }

    const size_t x = static_cast<size_t>(nx);
    const size_t y = static_cast<size_t>(ny);
    const size_t z = static_cast<size_t>(nz);
    if (x > std::numeric_limits<size_t>::max() / y
        || x * y > std::numeric_limits<size_t>::max() / z) {
        throw std::overflow_error("Grid dimensions exceed addressable storage");
    }
    return x * y * z;
}

int checkedMacDimension(int dimension) {
    if (dimension <= 0 || dimension == std::numeric_limits<int>::max()) {
        throw std::invalid_argument("MAC grid dimensions must be positive and non-overflowing");
    }
    return dimension;
}

void reportCudaCleanupError(cudaError_t result) noexcept {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed: %s\n", cudaGetErrorString(result));
    }
}

void write_to_log_cuda(const std::string& kernel_name, float ms) {
    std::ofstream log_file("profile_cuda.log", std::ios::app);
    log_file << "[CUDA Profile] Kernel: " << kernel_name 
             << " | Time: " << ms << " ms\n";
    // std::cout << "[LOG] " << kernel_name << " took " << ms << " ms\n";
}

void write_to_log(const std::string& kernel_name, float us) {
    std::ofstream log_file("profile.log", std::ios::app);
    log_file << "[Profile]: " << kernel_name 
             << " | Time: " << us << " μs\n";
    // std::cout << "[LOG] " << kernel_name << " took " << ms << " us\n";
}
