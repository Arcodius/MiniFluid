#pragma once
#include "grid.h"
#include "math.h"

class FluidSolver2D
{
private:
    int nx_;
    int ny_;
    float h_;
    float rho_;

    Grid2D density_next_;

public:
    Grid2D density;
    Grid2D divergence;
    MacGrid2D macgrid; // includes pressure


    FluidSolver2D(int nx, int ny, float h, float rho) 
        : nx_(nx), ny_(ny), h_(h), rho_(rho),
        density(nx, ny, h), divergence(nx, ny, h),
        macgrid(nx, ny, h, rho),
        density_next_(nx, ny, h)
    {
        initVelocity();
        initDensity();
    }

    void initDensity();
    void initVelocity();

    std::pair<int, int> getSize() const;

    void advectVelocity(float dt);
    void addForces(float dt);
    void enforceBoundaryVelocity();
    
    void computeDivergence();
    void solvePressure(float dt, int iterations);
    void applyPressureGradient(float dt);
    void testDivergence();

    void advectDensity(float dt);
    void step(float dt);
};
