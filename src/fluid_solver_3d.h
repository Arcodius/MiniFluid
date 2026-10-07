#pragma once
#include "grid.h"
#include "fluid_math.h"

class FluidSolver3D
{
private:
    int nx_;
    int ny_;
    int nz_;
    float h_;
    float rho_;

    Grid3D u_next_;
    Grid3D v_next_;
    Grid3D w_next_;

    Grid3D density_next_;
    Grid3D pressure_next_;

public:
    Grid3D density;
    Grid3D temperature;
    Grid3D divergence;
    MacGrid3D macgrid; // includes pressure


    FluidSolver3D(int nx, int ny, int nz, float h, float rho) 
        : nx_(nx), ny_(ny), nz_(nz), h_(h), rho_(rho),
        density(nx, ny, nz, h), temperature(nx, ny, nz, h), divergence(nx, ny, nz, h),
        macgrid(nx, ny, nz, h, rho),
        u_next_(nx + 1, ny, nz, h), v_next_(nx, ny + 1, nz, h), w_next_(nx, ny, nz + 1, h), density_next_(nx, ny, nz, h), pressure_next_(nx, ny, nz, h)
    {
        initVelocity();
        initDensity();
    }

    void initDensity();
    void initVelocity();

    Vec3 getSize() const { return Vec3(nx_, ny_, nz_); }

    void advectVelocity(float dt);
    void addForces(float dt);
    void addVelocitySource(float v0);
    void addDensitySource(float rho);
    void enforceBoundaryVelocity();
    
    void computeDivergence();
    void solvePressure(float dt, int iterations);
    void applyPressureGradient(float dt);
    void testDivergence();

    void advectDensity(float dt);
    void step(float dt);
};
