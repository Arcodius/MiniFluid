#include "grid.h"

struct  FluidSolver2D
{
    int nx;
    int ny;
    float dx;

    Grid2D density;
    Grid2D pressure;
    Grid2D divergence;

    Grid2D u;
    Grid2D v;
};
