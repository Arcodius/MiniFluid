#pragma once

#include "grid.h"

class CudaDivergence3D {
public:
    CudaDivergence3D(int nx, int ny, int nz);
    ~CudaDivergence3D();

    CudaDivergence3D(const CudaDivergence3D&) = delete;
    CudaDivergence3D& operator=(const CudaDivergence3D&) = delete;

    void compute(const MacGrid3D& macgrid, Grid3D& divergence);

private:
    int nx_;
    int ny_;
    int nz_;
    float* device_u_ = nullptr;
    float* device_v_ = nullptr;
    float* device_w_ = nullptr;
    float* device_divergence_ = nullptr;
};