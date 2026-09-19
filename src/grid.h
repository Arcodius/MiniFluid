#pragma once

#include <vector>

struct Vec2 {
    float x;
    float y;
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
    Vec2 sampleVelocity(float x, float y);
    Vec2 sampleVelocity(int i, int j);
    float divergenceAt(int i, int j);
    void applyPressureGradient(float dt);
};
