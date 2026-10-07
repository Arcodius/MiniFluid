#include "fluid_math.h"
#include "grid.h"

class CudaFluidSolver {
private:
    DeviceMacGrid3D velocity_;
public:
    CudaFluidSolver(int nx, int ny, int nz);
    ~CudaFluidSolver();

    void step(float dt);
};