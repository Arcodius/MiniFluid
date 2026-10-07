#pragma once

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
