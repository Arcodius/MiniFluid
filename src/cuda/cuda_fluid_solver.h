#include "fluid_math.h"
#include "grid.h"

class CudaFluidSolver {
private:
    DeviceMacGrid3D macgrid_;

    DeviceGrid3D u_next_;
    DeviceGrid3D v_next_;
    DeviceGrid3D w_next_;
    DeviceGrid3D density_;
    DeviceGrid3D divergence_;
    DeviceGrid3D density_next_;
    DeviceGrid3D pressure_next_;
public:
    CudaFluidSolver(int nx, int ny, int nz, float h, float rho);
    ~CudaFluidSolver();

    int nx() const noexcept { return macgrid_.nx(); }
    int ny() const noexcept { return macgrid_.ny(); }
    int nz() const noexcept { return macgrid_.nz(); }

    void copyDensitySlice(int k, float* destination, size_t element_count) const;
    void copyDivergenceSlice(int k, float* destination, size_t element_count) const;

    void initDensity();
    void initVelocity();

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