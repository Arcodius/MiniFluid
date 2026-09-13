#include "grid.h"

float& Grid2D::operator()(int i, int j) {
    return data[nx * j + i];
}

float Grid2D::operator()(int i, int j) const {
    return data[nx * j + i];
}