#pragma once

#include "grid.h"

#include <cstddef>
#include <cstdint>
#include <utility>

class Camera;
class CudaFluidSolver;

class CudaRenderer {
private:
    size_t width_ = 0;
    size_t height_ = 0;
    Camera& camera_;
    CudaFluidSolver& solver_;
    DeviceBuffer<uint8_t> devicePixels_;

public:
    explicit CudaRenderer(CudaFluidSolver& solver, Camera& camera, std::pair<size_t, size_t> resolution = {720, 720});

    void render(uint8_t* destination, size_t destinationSize);
    void setResolution(size_t w, size_t h);
    std::pair<size_t, size_t> getResolution() const;
};