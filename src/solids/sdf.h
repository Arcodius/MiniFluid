#pragma once

#include <cuda_runtime.h>
#include <cmath>

__host__ __device__ inline float3 operator-(float3 a, float3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__host__ __device__ inline float  length(float3 a) { return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); }

struct SphereSDF {
    float3 center;
    float radius;

    __host__ __device__ float phi(float3 p) const {
        return length(p - center) - radius;
    }
};

template <class Shape>
__host__ __device__ float3 sdfGradient(const Shape& s, float3 p, float eps) {
    float3 g = {
        s.phi({p.x+eps, p.y, p.z}) - s.phi({p.x-eps, p.y, p.z}),
        s.phi({p.x, p.y+eps, p.z}) - s.phi({p.x, p.y-eps, p.z}),
        s.phi({p.x, p.y, p.z+eps}) - s.phi({p.x, p.y, p.z-eps})
    };
    float len = length(g);
    return len > 1e-8f ? float3(g.x/len, g.y/len, g.z/len) : float3(0, 0, 0);
}

template <class Shape>
void bakeSDF(const Shape& s, Grid3D& phi, float3 origin, float h) {
    for (int k = 0; k < phi.depth();  ++k)
    for (int j = 0; j < phi.height(); ++j)
    for (int i = 0; i < phi.width();  ++i) {
        // Center sample
        float3 p = { origin.x + (i+0.5f)*h,
                     origin.y + (j+0.5f)*h,
                     origin.z + (k+0.5f)*h };
        phi(i,j,k) = s.phi(p);
    }
}
