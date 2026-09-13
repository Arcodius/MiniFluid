#include <vector>


struct Grid2D
{
    int nx, ny;
    float dx;

    std::vector<float> data;

    Grid2D() {}
    Grid2D(int nx, int ny, float dx): nx(nx), ny(ny), dx(dx), data(nx * ny, 0.0f) {}

    float& operator()(int i, int j);
    float operator()(int i, int j) const;

    
};
