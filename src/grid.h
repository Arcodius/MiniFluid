#pragma once

#include "cuda/cuda_check.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

struct Vec2 {
    float x;
    float y;
};

struct Vec3 {
    float x, y, z;

    explicit Vec3(float v) : x(v), y(v), z(v) {}
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};

class Grid2D {
private:
    int nx_, ny_;
    float h_;
    std::vector<float> data_;

public:
    Grid2D() {}
    Grid2D(int nx, int ny, float h): nx_(nx), ny_(ny), h_(h), data_(nx * ny, 0.0f) {}
    Grid2D(int nx, int ny, float h, float value): nx_(nx), ny_(ny), h_(h), data_(nx * ny, value) {}

    float& operator()(int i, int j);
    float operator()(int i, int j) const;
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }
    int width() const { return nx_; }
    int height() const { return ny_; }
    float spacing() const { return h_; }

    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }
};

class Grid3D {
private:
    int nx_, ny_, nz_;
    float h_;
    std::vector<float> data_;
public:
    Grid3D() {}
    Grid3D(int nx, int ny, int nz, float h): nx_(nx), ny_(ny), nz_(nz), h_(h), data_(nx * ny * nz, 0.0f) {}
    Grid3D(int nx, int ny, int nz, float h, float value): nx_(nx), ny_(ny), nz_(nz), h_(h), data_(nx * ny * nz, value) {}

    inline float& operator() (int i, int j, int k) {
        return data_[i + nx_ * (j + ny_ * k)];
    }

    inline float operator() (int i, int j, int k) const {
        return data_[i + nx_ * (j + ny_ * k)];
    }

    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }
    int width() const { return nx_; }
    int height() const { return ny_; }
    int depth() const { return nz_; }
    float spacing() const { return h_; }

    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }
};

class MacGrid2D {
private:
    int nx_, ny_;
    float h_;
    float rho_;
    Grid2D pressure_;
    Grid2D u_;
    Grid2D v_;

public:
    MacGrid2D(int nx, int ny, float h, float rho)
        : nx_(nx), ny_(ny), h_(h), rho_(rho),
          pressure_(nx, ny, h),
          u_(nx + 1, ny, h),
          v_(nx, ny + 1, h) {}

    Grid2D& pressure() { return pressure_; }
    const Grid2D& pressure() const { return pressure_; }
    Grid2D& u() { return u_; }
    const Grid2D& u() const { return u_; }
    Grid2D& v() { return v_; }
    const Grid2D& v() const { return v_; }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    float spacing() const { return h_; }
    float rho() const { return rho_; }

    Vec2 cellVelocity(int i, int j);
    Vec2 sampleVelocity(float x, float y) const;
    float divergenceAt(int i, int j);
    void applyPressureGradient(float dt);
};

class MacGrid3D {
private:
    int nx_, ny_, nz_;
    float h_;
    float rho_;
    Grid3D pressure_;
    Grid3D u_;
    Grid3D v_;
    Grid3D w_;

public:
    MacGrid3D(int nx, int ny, int nz, float h, float rho)
        : nx_(nx), ny_(ny), nz_(nz), h_(h), rho_(rho),
          pressure_(nx, ny, nz, h),
          u_(nx + 1, ny, nz, h),
          v_(nx, ny + 1, nz, h),
          w_(nx, ny, nz + 1, h) {}

    Grid3D& pressure() { return pressure_; }
    const Grid3D& pressure() const { return pressure_; }
    Grid3D& u() { return u_; }
    const Grid3D& u() const { return u_; }
    Grid3D& v() { return v_; }
    const Grid3D& v() const { return v_; }
    Grid3D& w() { return w_; }
    const Grid3D& w() const { return w_; }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    float spacing() const { return h_; }
    float rho() const { return rho_; }

    Vec3 cellVelocity(int i, int j, int k);
    Vec3 sampleVelocity(float x, float y, float z) const;
    float divergenceAt(int i, int j, int k);
    void applyPressureGradient(float dt);
};


template<class T>
struct GridView3D {
    T* data = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;

    GridView3D() = default;
    GridView3D(T* ptr, int x, int y, int z) 
        : data(ptr), nx(x), ny(y), nz(z) {}

    __host__ __device__ size_t size() const {
        return static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    }

    __host__ __device__ T& operator()(int x, int y, int z) const {
        const size_t index =
            (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y))
            * static_cast<size_t>(nx) + static_cast<size_t>(x);
        return data[index];
    }
};

template<class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t size) : size_(size) {
        static_assert(std::is_trivially_destructible_v<T>,
            "DeviceBuffer requires trivially destructible element types");
        if (size_ > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::overflow_error("Device buffer size exceeds addressable storage");
        }
        if (size_ != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() noexcept {
        releaseNoexcept();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            releaseNoexcept();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }

    void reset() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFree(data_));
            data_ = nullptr;
            size_ = 0;
        }
    }

private:
    void releaseNoexcept() noexcept {
        if (data_ != nullptr) {
            reportCudaCleanupError(cudaFree(data_));
            data_ = nullptr;
            size_ = 0;
        }
    }

    T* data_ = nullptr;
    size_t size_ = 0;
};

class DeviceGrid3D {
public:
    DeviceGrid3D() = default;
    DeviceGrid3D(int nx, int ny, int nz)
        : storage_(checkedElementCount(nx, ny, nz)),
          nx_(nx), ny_(ny), nz_(nz) {}

    DeviceGrid3D(const DeviceGrid3D&) = delete;
    DeviceGrid3D& operator=(const DeviceGrid3D&) = delete;
    DeviceGrid3D(DeviceGrid3D&&) noexcept = default;
    DeviceGrid3D& operator=(DeviceGrid3D&&) noexcept = default;

    GridView3D<float> view() noexcept {
        return GridView3D<float>(storage_.data(), nx_, ny_, nz_);
    }
    GridView3D<const float> view() const noexcept {
        return GridView3D<const float>(storage_.data(), nx_, ny_, nz_);
    }

    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int nz() const noexcept { return nz_; }
    size_t size() const noexcept { return storage_.size(); }

    void upload(const Grid3D& source) {
        validateDimensions(source);
        CUDA_CHECK(cudaMemcpy(storage_.data(), source.data().data(),
            size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void download(Grid3D& destination) const {
        validateDimensions(destination);
        CUDA_CHECK(cudaMemcpy(destination.data().data(), storage_.data(),
            size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

private:
    DeviceBuffer<float> storage_;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;

    void validateDimensions(const Grid3D& grid) const {
        if (grid.width() != nx_ || grid.height() != ny_ || grid.depth() != nz_) {
            throw std::invalid_argument("Host and device grid dimensions do not match");
        }
    }
};

class DeviceMacGrid3D {
public:
    DeviceMacGrid3D() = default;
    DeviceMacGrid3D(int nx, int ny, int nz, float h, float rho)
        : nx_(checkedMacDimension(nx)),
          ny_(checkedMacDimension(ny)),
          nz_(checkedMacDimension(nz)),
          h_(h), rho_(rho),
          pressure_(nx_, ny_, nz_),
          u_(nx_ + 1, ny_, nz_),
          v_(nx_, ny_ + 1, nz_),
          w_(nx_, ny_, nz_ + 1) {}

    DeviceMacGrid3D(const DeviceMacGrid3D&) = delete;
    DeviceMacGrid3D& operator=(const DeviceMacGrid3D&) = delete;
    DeviceMacGrid3D(DeviceMacGrid3D&&) noexcept = default;
    DeviceMacGrid3D& operator=(DeviceMacGrid3D&&) noexcept = default;

    DeviceGrid3D& pressure() noexcept { return pressure_; }
    const DeviceGrid3D& pressure() const noexcept { return pressure_; }
    DeviceGrid3D& u() noexcept { return u_; }
    const DeviceGrid3D& u() const noexcept { return u_; }
    DeviceGrid3D& v() noexcept { return v_; }
    const DeviceGrid3D& v() const noexcept { return v_; }
    DeviceGrid3D& w() noexcept { return w_; }
    const DeviceGrid3D& w() const noexcept { return w_; }

    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int nz() const noexcept { return nz_; }
    float spacing() const noexcept { return h_; }
    float rho() const noexcept { return rho_; }

    void upload(const MacGrid3D& source) {
        validateDimensions(source);
        pressure_.upload(source.pressure());
        u_.upload(source.u());
        v_.upload(source.v());
        w_.upload(source.w());
    }

    void download(MacGrid3D& destination) const {
        validateDimensions(destination);
        pressure_.download(destination.pressure());
        u_.download(destination.u());
        v_.download(destination.v());
        w_.download(destination.w());
    }

private:
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
    float h_ = 0.0f;
    float rho_ = 0.0f;
    DeviceGrid3D pressure_;
    DeviceGrid3D u_;
    DeviceGrid3D v_;
    DeviceGrid3D w_;

    void validateDimensions(const MacGrid3D& grid) const {
        if (grid.nx() != nx_ || grid.ny() != ny_ || grid.nz() != nz_) {
            throw std::invalid_argument("Host and device MAC grid dimensions do not match");
        }
    }
};