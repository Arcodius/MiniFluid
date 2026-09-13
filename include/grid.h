#include <vector>


struct Grid2D
{
    int nx, ny;
    float dx;

    std::vector<float> data;

    float& operator()(int i, int j);
    float operator()(int i, int j) const;
};
