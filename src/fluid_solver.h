#include "grid.h"
#include "math.h"

class FluidSolver2D
{
private:
    int nx_;
    int ny_;
    float dx_;

public:
    Grid2D density;
    Grid2D pressure;
    Grid2D divergence;

    Grid2D u;
    Grid2D v;

    FluidSolver2D(int nx, int ny, float dx) {
        nx_ = nx, ny_ = ny, dx_ = dx;

        density = Grid2D(nx, ny, dx);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float x = (i+0.5f) * dx;
                float y = (j+0.5f) * dx;

                float cx = 0.5f, cy = 0.5f;
                float r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);

                density(i, j) = exp(-100.f * r2);
            }
        }
    }

    std::pair<int, int> getSize() const {
        return std::pair<int, int>(nx_, ny_);
    }

    void step(float dt) {
        
    }
};
