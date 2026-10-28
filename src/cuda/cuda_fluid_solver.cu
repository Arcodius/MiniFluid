#include "cuda_fluid_solver.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr int kMaxEmitters = 4;

struct DeviceEmitter {
    float3 center;
    float radius;
    float verticalScale; // radius / thickness, or 1 for a sphere
    float3 velocity;
    float density;
    float temperature;
    float noiseAmplitude;
    float velocityJitter;
    float noiseFrequency;
    float strength;
};

struct DeviceEmitterSet {
    DeviceEmitter emitters[kMaxEmitters];
    int count;
    float time;
};

dim3 blocksFor(int nx, int ny, int nz, dim3 threads) {
    return dim3(
        (nx + threads.x - 1) / threads.x,
        (ny + threads.y - 1) / threads.y,
        (nz + threads.z - 1) / threads.z);
}

void clearDeviceGrid(DeviceGrid3D& grid) {
    const GridView3D<float> view = grid.view();
    CUDA_CHECK(cudaMemset(view.data, 0, view.size() * sizeof(float)));
}

template<class T>
void clearDeviceBuffer(DeviceBuffer<T>& buffer) {
    CUDA_CHECK(cudaMemset(buffer.data(), 0, buffer.size() * sizeof(T)));
}

void copySliceToHost(
    GridView3D<const float> grid,
    int k,
    float* destination,
    size_t element_count) {
    const size_t slice_size = static_cast<size_t>(grid.nx) * grid.ny;
    if (k < 0 || k >= grid.nz) {
        throw std::out_of_range("Slice index is outside the grid");
    }
    if (destination == nullptr || element_count != slice_size) {
        throw std::invalid_argument("Destination size does not match the grid slice");
    }

    const size_t offset = static_cast<size_t>(k) * slice_size;
    CUDA_CHECK(cudaMemcpy(
        destination, grid.data + offset, slice_size * sizeof(float),
        cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

__device__ float clampCoordinate(float coordinate, int extent) {
    return fminf(fmaxf(coordinate, 0.0f), static_cast<float>(extent - 1));
}

struct TrilinearCell {
    int x0, y0, z0, x1, y1, z1;
    float tx, ty, tz;
};

__device__ TrilinearCell locateCell(int nx, int ny, int nz, float x, float y, float z) {
    x = clampCoordinate(x, nx);
    y = clampCoordinate(y, ny);
    z = clampCoordinate(z, nz);

    TrilinearCell cell;
    cell.x0 = static_cast<int>(floorf(x));
    cell.y0 = static_cast<int>(floorf(y));
    cell.z0 = static_cast<int>(floorf(z));
    cell.x1 = min(cell.x0 + 1, nx - 1);
    cell.y1 = min(cell.y0 + 1, ny - 1);
    cell.z1 = min(cell.z0 + 1, nz - 1);
    cell.tx = x - static_cast<float>(cell.x0);
    cell.ty = y - static_cast<float>(cell.y0);
    cell.tz = z - static_cast<float>(cell.z0);
    return cell;
}

__device__ float sampleTrilinear(
    GridView3D<const float> grid, float x, float y, float z) {
    const TrilinearCell c = locateCell(grid.nx, grid.ny, grid.nz, x, y, z);

    const float x00 = grid(c.x0, c.y0, c.z0) + c.tx * (grid(c.x1, c.y0, c.z0) - grid(c.x0, c.y0, c.z0));
    const float x10 = grid(c.x0, c.y1, c.z0) + c.tx * (grid(c.x1, c.y1, c.z0) - grid(c.x0, c.y1, c.z0));
    const float x01 = grid(c.x0, c.y0, c.z1) + c.tx * (grid(c.x1, c.y0, c.z1) - grid(c.x0, c.y0, c.z1));
    const float x11 = grid(c.x0, c.y1, c.z1) + c.tx * (grid(c.x1, c.y1, c.z1) - grid(c.x0, c.y1, c.z1));
    const float y0_value = x00 + c.ty * (x10 - x00);
    const float y1_value = x01 + c.ty * (x11 - x01);
    return y0_value + c.tz * (y1_value - y0_value);
}

// Range of the eight samples used by sampleTrilinear; used as the MacCormack limiter.
__device__ void sampleRange(
    GridView3D<const float> grid, float x, float y, float z,
    float& min_value, float& max_value) {
    const TrilinearCell c = locateCell(grid.nx, grid.ny, grid.nz, x, y, z);
    const float values[8] = {
        grid(c.x0, c.y0, c.z0), grid(c.x1, c.y0, c.z0),
        grid(c.x0, c.y1, c.z0), grid(c.x1, c.y1, c.z0),
        grid(c.x0, c.y0, c.z1), grid(c.x1, c.y0, c.z1),
        grid(c.x0, c.y1, c.z1), grid(c.x1, c.y1, c.z1)};
    min_value = values[0];
    max_value = values[0];
    for (int n = 1; n < 8; ++n) {
        min_value = fminf(min_value, values[n]);
        max_value = fmaxf(max_value, values[n]);
    }
}

__device__ float3 sampleVelocity(
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float3 p,
    float inv_h) {
    return make_float3(
        sampleTrilinear(u, p.x * inv_h, p.y * inv_h - 0.5f, p.z * inv_h - 0.5f),
        sampleTrilinear(v, p.x * inv_h - 0.5f, p.y * inv_h, p.z * inv_h - 0.5f),
        sampleTrilinear(w, p.x * inv_h - 0.5f, p.y * inv_h - 0.5f, p.z * inv_h));
}

// Second-order (midpoint) backtrace of the characteristic through p.
__device__ float3 backtrace(
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float3 p,
    float dt,
    float inv_h) {
    const float3 v1 = sampleVelocity(u, v, w, p, inv_h);
    const float3 mid = make_float3(
        p.x - 0.5f * dt * v1.x, p.y - 0.5f * dt * v1.y, p.z - 0.5f * dt * v1.z);
    const float3 v2 = sampleVelocity(u, v, w, mid, inv_h);
    return make_float3(p.x - dt * v2.x, p.y - dt * v2.y, p.z - dt * v2.z);
}

// ---------------------------------------------------------------------------
// Noise used by the emitters
// ---------------------------------------------------------------------------

__device__ float hashLattice(int x, int y, int z) {
    unsigned int h = static_cast<unsigned int>(x) * 73856093u
        ^ static_cast<unsigned int>(y) * 19349663u
        ^ static_cast<unsigned int>(z) * 83492791u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return static_cast<float>(h) * (2.0f / 4294967295.0f) - 1.0f;
}

__device__ float valueNoise(float3 p) {
    const float fx = floorf(p.x);
    const float fy = floorf(p.y);
    const float fz = floorf(p.z);
    const int x = static_cast<int>(fx);
    const int y = static_cast<int>(fy);
    const int z = static_cast<int>(fz);
    float tx = p.x - fx;
    float ty = p.y - fy;
    float tz = p.z - fz;
    tx = tx * tx * (3.0f - 2.0f * tx);
    ty = ty * ty * (3.0f - 2.0f * ty);
    tz = tz * tz * (3.0f - 2.0f * tz);

    const float x00 = hashLattice(x, y, z) + tx * (hashLattice(x + 1, y, z) - hashLattice(x, y, z));
    const float x10 = hashLattice(x, y + 1, z) + tx * (hashLattice(x + 1, y + 1, z) - hashLattice(x, y + 1, z));
    const float x01 = hashLattice(x, y, z + 1) + tx * (hashLattice(x + 1, y, z + 1) - hashLattice(x, y, z + 1));
    const float x11 = hashLattice(x, y + 1, z + 1) + tx * (hashLattice(x + 1, y + 1, z + 1) - hashLattice(x, y + 1, z + 1));
    const float y0_value = x00 + ty * (x10 - x00);
    const float y1_value = x01 + ty * (x11 - x01);
    return y0_value + tz * (y1_value - y0_value);
}

// Two octaves of value noise, roughly in [-1, 1]. Higher octaves would inject grid-scale
// noise that MacCormack advection preserves and the renderer shows as grain.
__device__ float fbm(float3 p) {
    float sum = 0.0f;
    float amplitude = 0.65f;
    for (int octave = 0; octave < 2; ++octave) {
        sum += amplitude * valueNoise(p);
        p = make_float3(p.x * 2.03f + 17.1f, p.y * 2.03f - 7.3f, p.z * 2.03f + 3.7f);
        amplitude *= 0.5f;
    }
    return sum;
}

__device__ float emitterFalloff(const DeviceEmitter& e, float3 p) {
    const float dx = p.x - e.center.x;
    const float dy = (p.y - e.center.y) * e.verticalScale;
    const float dz = p.z - e.center.z;
    const float distance = sqrtf(dx * dx + dy * dy + dz * dz) / e.radius;
    const float t = fminf(fmaxf((1.0f - distance) * 2.0f, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ float3 emitterNoiseCoordinate(const DeviceEmitter& e, float3 p, float time) {
    return make_float3(
        p.x * e.noiseFrequency + 0.37f * time,
        p.y * e.noiseFrequency - 1.9f * time,
        p.z * e.noiseFrequency + 0.53f * time);
}

// ---------------------------------------------------------------------------
// Advection
// ---------------------------------------------------------------------------

// Fields that share sample positions are advected together so that they share the backtrace.
template<int N>
struct AdvectFields {
    GridView3D<const float> source[N];
    GridView3D<float> destination[N];
    float decay[N];
};

template<int N>
struct CorrectFields {
    GridView3D<const float> original[N];
    GridView3D<const float> forward[N];
    GridView3D<float> result[N];  // Holds the backward estimate on entry
    float decay[N];
};

// Semi-Lagrangian advection of fields whose sample (i, j, k) sits at ((i, j, k) + offset) * h.
template<int N>
__global__ void advectKernel(
    AdvectFields<N> fields,
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float3 offset,
    float h,
    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const GridView3D<float>& extent = fields.destination[0];
    if (i >= extent.nx || j >= extent.ny || k >= extent.nz) {
        return;
    }

    const float inv_h = 1.0f / h;
    const float3 p = make_float3(
        (static_cast<float>(i) + offset.x) * h,
        (static_cast<float>(j) + offset.y) * h,
        (static_cast<float>(k) + offset.z) * h);
    const float3 departure = backtrace(u, v, w, p, dt, inv_h);
    for (int n = 0; n < N; ++n) {
        fields.destination[n](i, j, k) = fields.decay[n] * sampleTrilinear(
            fields.source[n],
            departure.x * inv_h - offset.x,
            departure.y * inv_h - offset.y,
            departure.z * inv_h - offset.z);
    }
}

// MacCormack correction: result = forward + (original - backward) / 2, clamped to the
// range of the values the forward step interpolated from.
template<int N>
__global__ void maccormackCorrectKernel(
    CorrectFields<N> fields,
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float3 offset,
    float h,
    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const GridView3D<float>& extent = fields.result[0];
    if (i >= extent.nx || j >= extent.ny || k >= extent.nz) {
        return;
    }

    const float inv_h = 1.0f / h;
    const float3 p = make_float3(
        (static_cast<float>(i) + offset.x) * h,
        (static_cast<float>(j) + offset.y) * h,
        (static_cast<float>(k) + offset.z) * h);
    const float3 departure = backtrace(u, v, w, p, dt, inv_h);
    for (int n = 0; n < N; ++n) {
        float min_value;
        float max_value;
        sampleRange(fields.original[n],
            departure.x * inv_h - offset.x,
            departure.y * inv_h - offset.y,
            departure.z * inv_h - offset.z,
            min_value, max_value);

        const float corrected = fields.forward[n](i, j, k)
            + 0.5f * (fields.original[n](i, j, k) - fields.result[n](i, j, k));
        fields.result[n](i, j, k) = fields.decay[n] * fminf(fmaxf(corrected, min_value), max_value);
    }
}

template<int N>
void launchAdvect(
    const AdvectFields<N>& fields,
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float3 offset, float h, float dt) {
    const dim3 threads(8, 8, 4);
    const GridView3D<float>& extent = fields.destination[0];
    advectKernel<N><<<blocksFor(extent.nx, extent.ny, extent.nz, threads), threads>>>(
        fields, u, v, w, offset, h, dt);
    CUDA_CHECK(cudaGetLastError());
}

template<int N>
void launchMaccormackCorrect(
    const CorrectFields<N>& fields,
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float3 offset, float h, float dt) {
    const dim3 threads(8, 8, 4);
    const GridView3D<float>& extent = fields.result[0];
    maccormackCorrectKernel<N><<<blocksFor(extent.nx, extent.ny, extent.nz, threads), threads>>>(
        fields, u, v, w, offset, h, dt);
    CUDA_CHECK(cudaGetLastError());
}

GridView3D<const float> constView(DeviceGrid3D& grid) {
    return static_cast<const DeviceGrid3D&>(grid).view();
}

// ---------------------------------------------------------------------------
// Sources and forces
// ---------------------------------------------------------------------------

// Launched over (nx + 1) x (ny + 1) x (nz + 1) so each thread can update one cell and the
// three faces on its lower side.
__global__ void addSourcesKernel(
    GridView3D<float> density,
    GridView3D<float> temperature,
    GridView3D<float> u,
    GridView3D<float> v,
    GridView3D<float> w,
    DeviceEmitterSet set,
    float h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const int nx = density.nx;
    const int ny = density.ny;
    const int nz = density.nz;
    if (i > nx || j > ny || k > nz) {
        return;
    }

    const float x = static_cast<float>(i) * h;
    const float y = static_cast<float>(j) * h;
    const float z = static_cast<float>(k) * h;
    const float half_h = 0.5f * h;

    for (int n = 0; n < set.count; ++n) {
        const DeviceEmitter& e = set.emitters[n];
        if (e.strength <= 0.0f) {
            continue;
        }

        if (i < nx && j < ny && k < nz) {
            const float3 p = make_float3(x + half_h, y + half_h, z + half_h);
            const float falloff = emitterFalloff(e, p) * e.strength;
            if (falloff > 0.0f) {
                const float noise = fbm(emitterNoiseCoordinate(e, p, set.time));
                const float modulation = fmaxf(1.0f + e.noiseAmplitude * noise, 0.0f);
                density(i, j, k) = fmaxf(density(i, j, k), e.density * falloff * modulation);
                temperature(i, j, k) = fmaxf(temperature(i, j, k), e.temperature * falloff * modulation);
            }
        }

        // Face velocities relax toward the emitter velocity plus a noisy perturbation that
        // breaks the symmetry of the plume.
        if (j < ny && k < nz) {
            const float3 p = make_float3(x, y + half_h, z + half_h);
            const float falloff = emitterFalloff(e, p) * e.strength;
            if (falloff > 0.0f) {
                const float3 q = emitterNoiseCoordinate(e, p, set.time);
                const float target = e.velocity.x
                    + e.velocityJitter * fbm(make_float3(q.x + 31.4f, q.y, q.z));
                u(i, j, k) += falloff * (target - u(i, j, k));
            }
        }
        if (i < nx && k < nz) {
            const float3 p = make_float3(x + half_h, y, z + half_h);
            const float falloff = emitterFalloff(e, p) * e.strength;
            if (falloff > 0.0f) {
                const float3 q = emitterNoiseCoordinate(e, p, set.time);
                const float target = e.velocity.y
                    + e.velocityJitter * fbm(make_float3(q.x, q.y + 47.2f, q.z));
                v(i, j, k) += falloff * (target - v(i, j, k));
            }
        }
        if (i < nx && j < ny) {
            const float3 p = make_float3(x + half_h, y + half_h, z);
            const float falloff = emitterFalloff(e, p) * e.strength;
            if (falloff > 0.0f) {
                const float3 q = emitterNoiseCoordinate(e, p, set.time);
                const float target = e.velocity.z
                    + e.velocityJitter * fbm(make_float3(q.x, q.y, q.z + 59.8f));
                w(i, j, k) += falloff * (target - w(i, j, k));
            }
        }
    }
}

__device__ float3 cellVelocity(
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    int i, int j, int k) {
    return make_float3(
        0.5f * (u(i, j, k) + u(i + 1, j, k)),
        0.5f * (v(i, j, k) + v(i, j + 1, k)),
        0.5f * (w(i, j, k) + w(i, j, k + 1)));
}

__global__ void vorticityKernel(
    GridView3D<float4> vorticity,
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float inv_2h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= vorticity.nx || j >= vorticity.ny || k >= vorticity.nz) {
        return;
    }

    const float3 left = cellVelocity(u, v, w, max(i - 1, 0), j, k);
    const float3 right = cellVelocity(u, v, w, min(i + 1, vorticity.nx - 1), j, k);
    const float3 down = cellVelocity(u, v, w, i, max(j - 1, 0), k);
    const float3 up = cellVelocity(u, v, w, i, min(j + 1, vorticity.ny - 1), k);
    const float3 back = cellVelocity(u, v, w, i, j, max(k - 1, 0));
    const float3 front = cellVelocity(u, v, w, i, j, min(k + 1, vorticity.nz - 1));

    const float wx = ((up.z - down.z) - (front.y - back.y)) * inv_2h;
    const float wy = ((front.x - back.x) - (right.z - left.z)) * inv_2h;
    const float wz = ((right.y - left.y) - (up.x - down.x)) * inv_2h;
    vorticity(i, j, k) = make_float4(wx, wy, wz, sqrtf(wx * wx + wy * wy + wz * wz));
}

// Cell-centered body force: vorticity confinement, buoyancy, and wind.
__global__ void cellForceKernel(
    GridView3D<float4> force,
    GridView3D<const float4> vorticity,
    GridView3D<const float> density,
    GridView3D<const float> temperature,
    float h,
    float confinement,
    float alpha,
    float beta,
    float3 wind) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= force.nx || j >= force.ny || k >= force.nz) {
        return;
    }

    const float4 omega = vorticity(i, j, k);
    const float gx = vorticity(min(i + 1, force.nx - 1), j, k).w - vorticity(max(i - 1, 0), j, k).w;
    const float gy = vorticity(i, min(j + 1, force.ny - 1), k).w - vorticity(i, max(j - 1, 0), k).w;
    const float gz = vorticity(i, j, min(k + 1, force.nz - 1)).w - vorticity(i, j, max(k - 1, 0)).w;
    const float inv_length = 1.0f / (sqrtf(gx * gx + gy * gy + gz * gz) + 1e-6f);
    const float normal_x = gx * inv_length;
    const float normal_y = gy * inv_length;
    const float normal_z = gz * inv_length;

    const float scale = confinement * h;
    const float fx = scale * (normal_y * omega.z - normal_z * omega.y) + wind.x;
    float fy = scale * (normal_z * omega.x - normal_x * omega.z) + wind.y;
    const float fz = scale * (normal_x * omega.y - normal_y * omega.x) + wind.z;
    fy += beta * temperature(i, j, k) - alpha * density(i, j, k);
    force(i, j, k) = make_float4(fx, fy, fz, 0.0f);
}

template<int Component>
__global__ void applyFaceForceKernel(
    GridView3D<float> velocity,
    GridView3D<const float4> force,
    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= velocity.nx || j >= velocity.ny || k >= velocity.nz) {
        return;
    }

    float face_force;
    if constexpr (Component == 0) {
        if (i == 0 || i >= force.nx) return;
        face_force = 0.5f * (force(i - 1, j, k).x + force(i, j, k).x);
    } else if constexpr (Component == 1) {
        if (j == 0 || j >= force.ny) return;
        face_force = 0.5f * (force(i, j - 1, k).y + force(i, j, k).y);
    } else {
        if (k == 0 || k >= force.nz) return;
        face_force = 0.5f * (force(i, j, k - 1).z + force(i, j, k).z);
    }
    velocity(i, j, k) += dt * face_force;
}

template<int Component>
void launchApplyFaceForce(
    GridView3D<float> velocity, GridView3D<const float4> force, float dt) {
    const dim3 threads(8, 8, 4);
    applyFaceForceKernel<Component><<<blocksFor(velocity.nx, velocity.ny, velocity.nz, threads), threads>>>(
        velocity, force, dt);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Boundaries and projection
// ---------------------------------------------------------------------------

__global__ void zeroUBoundaryKernel(GridView3D<float> u, int nx) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < u.ny && k < u.nz) {
        u(0, j, k) = 0.0f;
        u(nx, j, k) = 0.0f;
    }
}

__global__ void zeroVBoundaryKernel(GridView3D<float> v, int ny, int open_top) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < v.nx && k < v.nz) {
        v(i, 0, k) = 0.0f;
        if (!open_top) {
            v(i, ny, k) = 0.0f;
        }
    }
}

__global__ void zeroWBoundaryKernel(GridView3D<float> w, int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < w.nx && j < w.ny) {
        w(i, j, 0) = 0.0f;
        w(i, j, nz) = 0.0f;
    }
}

__global__ void computeDivergenceKernel(
    GridView3D<float> d,
    GridView3D<const float> u,
    GridView3D<const float> v,
    GridView3D<const float> w,
    float h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= d.nx || j >= d.ny || k >= d.nz) return;

    const float du = u(i + 1, j, k) - u(i, j, k);
    const float dv = v(i, j + 1, k) - v(i, j, k);
    const float dw = w(i, j, k + 1) - w(i, j, k);
    d(i, j, k) = (du + dv + dw) / h;
}

// The pressure Poisson equation is solved with a cell-centered geometric multigrid V-cycle.
// Every level solves (sum of neighbors - count * p) = rhs_scale * rhs, where count is the
// number of non-wall neighbors. Walls are Neumann boundaries (the neighbor is dropped); open
// faces are Dirichlet boundaries with zero pressure that add their weight to count.
// On the finest level the zero-pressure ghost sits one cell outside the domain, so the
// weight is 1; coarse levels keep that boundary at the same physical location.
// The floor (y = 0) is always a wall.
struct BoundaryWeights {
    float top;   // 0 for a closed top
    float side;  // 0 for closed x and z faces
};

struct PoissonNeighbors {
    float sum;
    float count;
};

template<class T>
__device__ PoissonNeighbors gatherNeighbors(
    GridView3D<T> p, int i, int j, int k, BoundaryWeights weights) {
    PoissonNeighbors n{0.0f, 0.0f};
    if (i > 0) { n.sum += p(i - 1, j, k); n.count += 1.0f; }
    else { n.count += weights.side; }
    if (i < p.nx - 1) { n.sum += p(i + 1, j, k); n.count += 1.0f; }
    else { n.count += weights.side; }
    if (j > 0) { n.sum += p(i, j - 1, k); n.count += 1.0f; }
    if (j < p.ny - 1) { n.sum += p(i, j + 1, k); n.count += 1.0f; }
    else { n.count += weights.top; }
    if (k > 0) { n.sum += p(i, j, k - 1); n.count += 1.0f; }
    else { n.count += weights.side; }
    if (k < p.nz - 1) { n.sum += p(i, j, k + 1); n.count += 1.0f; }
    else { n.count += weights.side; }
    return n;
}

// One half-sweep of red-black Gauss-Seidel with over-relaxation factor omega.
__global__ void redBlackSorKernel(
    GridView3D<float> pressure,
    GridView3D<const float> rhs,
    float rhs_scale,
    float omega,
    int parity,
    BoundaryWeights weights) {
    const int half_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const int i = 2 * half_i + ((j + k + parity) & 1);
    if (i >= pressure.nx || j >= pressure.ny || k >= pressure.nz) {
        return;
    }

    const PoissonNeighbors n = gatherNeighbors(pressure, i, j, k, weights);
    const float gauss_seidel = (n.sum - rhs_scale * rhs(i, j, k)) / n.count;
    const float current = pressure(i, j, k);
    pressure(i, j, k) = current + omega * (gauss_seidel - current);
}

// residual = rhs_scale * rhs - (sum - count * p), in the units of rhs_scale * rhs.
__global__ void residualKernel(
    GridView3D<float> residual,
    GridView3D<const float> pressure,
    GridView3D<const float> rhs,
    float rhs_scale,
    BoundaryWeights weights) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= residual.nx || j >= residual.ny || k >= residual.nz) {
        return;
    }

    const PoissonNeighbors n = gatherNeighbors(pressure, i, j, k, weights);
    residual(i, j, k) = rhs_scale * rhs(i, j, k) - (n.sum - n.count * pressure(i, j, k));
}

// Averages 2x2x2 fine residuals into the coarse right-hand side and clears the coarse
// correction. Since the coarse spacing is twice the fine spacing, the coarse level uses
// rhs_scale = 4 with this right-hand side.
__global__ void restrictKernel(
    GridView3D<float> coarse_rhs,
    GridView3D<float> coarse_pressure,
    GridView3D<const float> fine_residual) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= coarse_rhs.nx || j >= coarse_rhs.ny || k >= coarse_rhs.nz) {
        return;
    }

    float sum = 0.0f;
    for (int dk = 0; dk < 2; ++dk) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int di = 0; di < 2; ++di) {
                sum += fine_residual(2 * i + di, 2 * j + dj, 2 * k + dk);
            }
        }
    }
    coarse_rhs(i, j, k) = 0.125f * sum;
    coarse_pressure(i, j, k) = 0.0f;
}

// Adds the trilinearly interpolated coarse correction to the fine solution.
__global__ void prolongateKernel(
    GridView3D<float> fine_pressure,
    GridView3D<const float> coarse_pressure) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= fine_pressure.nx || j >= fine_pressure.ny || k >= fine_pressure.nz) {
        return;
    }

    fine_pressure(i, j, k) += sampleTrilinear(coarse_pressure,
        0.5f * (static_cast<float>(i) + 0.5f) - 0.5f,
        0.5f * (static_cast<float>(j) + 0.5f) - 0.5f,
        0.5f * (static_cast<float>(k) + 0.5f) - 0.5f);
}

struct PoissonLevel {
    GridView3D<float> pressure;
    GridView3D<const float> rhs;
    GridView3D<float> rhs_storage;  // Written by restriction; empty on the finest level
    GridView3D<float> residual;
    float rhs_scale;
    BoundaryWeights weights;
};

void smooth(const PoissonLevel& level, int iterations, float omega) {
    const dim3 threads(8, 8, 4);
    const GridView3D<float>& p = level.pressure;
    const dim3 blocks = blocksFor((p.nx + 1) / 2, p.ny, p.nz, threads);
    for (int iter = 0; iter < iterations; ++iter) {
        for (int parity = 0; parity < 2; ++parity) {
            redBlackSorKernel<<<blocks, threads>>>(
                p, level.rhs, level.rhs_scale, omega, parity, level.weights);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

void vCycle(const std::vector<PoissonLevel>& levels, size_t index) {
    constexpr int pre_smoothing = 2;
    constexpr int post_smoothing = 2;
    constexpr int coarsest_iterations = 24;
    const PoissonLevel& level = levels[index];
    if (index + 1 == levels.size()) {
        smooth(level, coarsest_iterations, 1.5f);
        return;
    }

    smooth(level, pre_smoothing, 1.0f);

    const dim3 threads(8, 8, 4);
    const GridView3D<float>& p = level.pressure;
    residualKernel<<<blocksFor(p.nx, p.ny, p.nz, threads), threads>>>(
        level.residual, GridView3D<const float>(p.data, p.nx, p.ny, p.nz),
        level.rhs, level.rhs_scale, level.weights);
    CUDA_CHECK(cudaGetLastError());

    const PoissonLevel& coarse = levels[index + 1];
    const GridView3D<float>& cp = coarse.pressure;
    restrictKernel<<<blocksFor(cp.nx, cp.ny, cp.nz, threads), threads>>>(
        coarse.rhs_storage, cp,
        GridView3D<const float>(level.residual.data, p.nx, p.ny, p.nz));
    CUDA_CHECK(cudaGetLastError());

    vCycle(levels, index + 1);

    prolongateKernel<<<blocksFor(p.nx, p.ny, p.nz, threads), threads>>>(
        p, GridView3D<const float>(cp.data, cp.nx, cp.ny, cp.nz));
    CUDA_CHECK(cudaGetLastError());

    smooth(level, post_smoothing, 1.0f);
}

template<int Component>
__global__ void pressureGradientKernel(
    GridView3D<float> velocity,
    GridView3D<const float> pressure,
    float inv_rho,
    float inv_h,
    float dt,
    int open_top,
    int open_sides) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= velocity.nx || j >= velocity.ny || k >= velocity.nz) {
        return;
    }

    // Faces on open boundaries see zero pressure outside the domain.
    float pressure_difference;
    if constexpr (Component == 0) {
        if ((i == 0 || i >= pressure.nx) && !open_sides) return;
        const float upper = i < pressure.nx ? pressure(i, j, k) : 0.0f;
        const float lower = i > 0 ? pressure(i - 1, j, k) : 0.0f;
        pressure_difference = upper - lower;
    } else if constexpr (Component == 1) {
        if (j == 0) return;
        if (j >= pressure.ny) {
            if (!open_top) return;
            pressure_difference = -pressure(i, j - 1, k);
        } else {
            pressure_difference = pressure(i, j, k) - pressure(i, j - 1, k);
        }
    } else {
        if ((k == 0 || k >= pressure.nz) && !open_sides) return;
        const float upper = k < pressure.nz ? pressure(i, j, k) : 0.0f;
        const float lower = k > 0 ? pressure(i, j, k - 1) : 0.0f;
        pressure_difference = upper - lower;
    }

    velocity(i, j, k) -= dt * inv_rho * pressure_difference * inv_h;
}

template<int Component>
void launchPressureGradientKernel(
    GridView3D<float> velocity,
    GridView3D<const float> pressure,
    float inv_rho, float inv_h, float dt, int open_top, int open_sides) {
    const dim3 threads(8, 8, 4);
    pressureGradientKernel<Component><<<blocksFor(velocity.nx, velocity.ny, velocity.nz, threads), threads>>>(
        velocity, pressure, inv_rho, inv_h, dt, open_top, open_sides);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

CudaFluidSolver::CudaFluidSolver(int nx, int ny, int nz, float h, float rho)
    : macgrid_(nx, ny, nz, h, rho),
      u_next_(nx + 1, ny, nz),
      v_next_(nx, ny + 1, nz),
      w_next_(nx, ny, nz + 1),
      u_scratch_(nx + 1, ny, nz),
      v_scratch_(nx, ny + 1, nz),
      w_scratch_(nx, ny, nz + 1),
      density_(nx, ny, nz),
      temperature_(nx, ny, nz),
      density_next_(nx, ny, nz),
      temperature_next_(nx, ny, nz),
      density_scratch_(nx, ny, nz),
      temperature_scratch_(nx, ny, nz),
      divergence_(nx, ny, nz),
      residual_(nx, ny, nz),
      vorticity_(checkedElementCount(nx, ny, nz)),
      cell_force_(checkedElementCount(nx, ny, nz)) {
    // Coarsen while every dimension stays even and the grid is not yet tiny.
    int level_nx = nx;
    int level_ny = ny;
    int level_nz = nz;
    while (level_nx % 2 == 0 && level_ny % 2 == 0 && level_nz % 2 == 0
           && std::min({level_nx, level_ny, level_nz}) >= 8) {
        level_nx /= 2;
        level_ny /= 2;
        level_nz /= 2;
        coarse_levels_.push_back(MultigridLevel{
            DeviceGrid3D(level_nx, level_ny, level_nz),
            DeviceGrid3D(level_nx, level_ny, level_nz),
            DeviceGrid3D(level_nx, level_ny, level_nz)});
    }

    emitters_.push_back(SmokeEmitter{});
    reset();
}

CudaFluidSolver::~CudaFluidSolver() = default;

void CudaFluidSolver::copyDensitySlice(
    int k, float* destination, size_t element_count) const {
    copySliceToHost(density_.view(), k, destination, element_count);
}

void CudaFluidSolver::copyDivergenceSlice(
    int k, float* destination, size_t element_count) const {
    copySliceToHost(divergence_.view(), k, destination, element_count);
}

void CudaFluidSolver::reset() {
    initDensity();
    initVelocity();
    clearDeviceGrid(macgrid_.pressure());
    clearDeviceGrid(divergence_);
    clearDeviceBuffer(vorticity_);
    clearDeviceBuffer(cell_force_);
    time_ = 0.0f;
}

void CudaFluidSolver::initDensity() {
    clearDeviceGrid(density_);
    clearDeviceGrid(temperature_);
}

void CudaFluidSolver::initVelocity() {
    clearDeviceGrid(macgrid_.u());
    clearDeviceGrid(macgrid_.v());
    clearDeviceGrid(macgrid_.w());
}

void CudaFluidSolver::advectVelocity(float dt) {
    const float h = macgrid_.spacing();
    const auto u = constView(macgrid_.u());
    const auto v = constView(macgrid_.v());
    const auto w = constView(macgrid_.w());
    const float decay = std::exp(-params_.velocityDamping * dt);
    const float3 offsets[3] = {
        make_float3(0.0f, 0.5f, 0.5f),
        make_float3(0.5f, 0.0f, 0.5f),
        make_float3(0.5f, 0.5f, 0.0f)};
    const GridView3D<const float> originals[3] = {u, v, w};
    DeviceGrid3D* forwards[3] = {&u_next_, &v_next_, &w_next_};
    DeviceGrid3D* results[3] = {&u_scratch_, &v_scratch_, &w_scratch_};

    if (!params_.maccormackVelocity) {
        for (int c = 0; c < 3; ++c) {
            launchAdvect<1>({{originals[c]}, {forwards[c]->view()}, {decay}},
                u, v, w, offsets[c], h, dt);
        }
        std::swap(macgrid_.u(), u_next_);
        std::swap(macgrid_.v(), v_next_);
        std::swap(macgrid_.w(), w_next_);
        return;
    }

    for (int c = 0; c < 3; ++c) {
        launchAdvect<1>({{originals[c]}, {forwards[c]->view()}, {1.0f}},
            u, v, w, offsets[c], h, dt);
    }
    for (int c = 0; c < 3; ++c) {
        launchAdvect<1>({{constView(*forwards[c])}, {results[c]->view()}, {1.0f}},
            u, v, w, offsets[c], h, -dt);
    }
    for (int c = 0; c < 3; ++c) {
        launchMaccormackCorrect<1>(
            {{originals[c]}, {constView(*forwards[c])}, {results[c]->view()}, {decay}},
            u, v, w, offsets[c], h, dt);
    }

    std::swap(macgrid_.u(), u_scratch_);
    std::swap(macgrid_.v(), v_scratch_);
    std::swap(macgrid_.w(), w_scratch_);
}

void CudaFluidSolver::addSources(float dt) {
    DeviceEmitterSet set{};
    set.time = time_;
    set.count = std::min(static_cast<int>(emitters_.size()), kMaxEmitters);
    for (int n = 0; n < set.count; ++n) {
        const SmokeEmitter& e = emitters_[n];
        DeviceEmitter& d = set.emitters[n];
        const float angle = e.orbitSpeed * time_;
        d.center = make_float3(
            e.center.x + e.orbitRadius * std::cos(angle),
            e.center.y,
            e.center.z + e.orbitRadius * std::sin(angle));
        d.radius = e.radius;
        d.verticalScale = e.thickness > 0.0f ? e.radius / e.thickness : 1.0f;
        d.velocity = e.velocity;
        d.density = e.density;
        d.temperature = e.temperature;
        d.noiseAmplitude = e.noiseAmplitude;
        d.velocityJitter = e.velocityJitter;
        d.noiseFrequency = e.noiseFrequency;
        d.strength = 1.0f;
        if (e.pulsePeriod > 0.0f) {
            const float phase = std::fmod(time_, e.pulsePeriod) / e.pulsePeriod;
            d.strength = phase < e.pulseDuty ? 1.0f : 0.0f;
        }
    }

    const dim3 threads(8, 8, 4);
    const int nx = macgrid_.nx();
    const int ny = macgrid_.ny();
    const int nz = macgrid_.nz();
    addSourcesKernel<<<blocksFor(nx + 1, ny + 1, nz + 1, threads), threads>>>(
        density_.view(), temperature_.view(),
        macgrid_.u().view(), macgrid_.v().view(), macgrid_.w().view(),
        set, macgrid_.spacing());
    CUDA_CHECK(cudaGetLastError());
}

void CudaFluidSolver::addForces(float dt) {
    const int nx = macgrid_.nx();
    const int ny = macgrid_.ny();
    const int nz = macgrid_.nz();
    const float h = macgrid_.spacing();
    const dim3 threads(8, 8, 4);
    const dim3 blocks = blocksFor(nx, ny, nz, threads);
    const GridView3D<float4> vorticity(vorticity_.data(), nx, ny, nz);
    const GridView3D<float4> force(cell_force_.data(), nx, ny, nz);

    vorticityKernel<<<blocks, threads>>>(
        vorticity,
        constView(macgrid_.u()), constView(macgrid_.v()), constView(macgrid_.w()),
        0.5f / h);
    CUDA_CHECK(cudaGetLastError());

    cellForceKernel<<<blocks, threads>>>(
        force,
        GridView3D<const float4>(vorticity_.data(), nx, ny, nz),
        constView(density_), constView(temperature_),
        h, params_.vorticity,
        params_.buoyancyDensity, params_.buoyancyTemperature, params_.wind);
    CUDA_CHECK(cudaGetLastError());

    const GridView3D<const float4> const_force(cell_force_.data(), nx, ny, nz);
    launchApplyFaceForce<0>(macgrid_.u().view(), const_force, dt);
    launchApplyFaceForce<1>(macgrid_.v().view(), const_force, dt);
    launchApplyFaceForce<2>(macgrid_.w().view(), const_force, dt);
}

void CudaFluidSolver::enforceBoundaryVelocity() {
    constexpr int tile = 16;
    const dim3 threads(tile, tile);

    const GridView3D<float> u = macgrid_.u().view();
    if (!params_.openSides) {
        const dim3 u_blocks(
            (u.ny + tile - 1) / tile,
            (u.nz + tile - 1) / tile);
        zeroUBoundaryKernel<<<u_blocks, threads>>>(u, macgrid_.nx());
        CUDA_CHECK(cudaGetLastError());
    }

    const GridView3D<float> v = macgrid_.v().view();
    const dim3 v_blocks(
        (v.nx + tile - 1) / tile,
        (v.nz + tile - 1) / tile);
    zeroVBoundaryKernel<<<v_blocks, threads>>>(v, macgrid_.ny(), params_.openTop ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());

    const GridView3D<float> w = macgrid_.w().view();
    if (!params_.openSides) {
        const dim3 w_blocks(
            (w.nx + tile - 1) / tile,
            (w.ny + tile - 1) / tile);
        zeroWBoundaryKernel<<<w_blocks, threads>>>(w, macgrid_.nz());
        CUDA_CHECK(cudaGetLastError());
    }
}

void CudaFluidSolver::computeDivergence() {
    const dim3 threads(8, 8, 4);
    const GridView3D<float> d = divergence_.view();
    computeDivergenceKernel<<<blocksFor(d.nx, d.ny, d.nz, threads), threads>>>(
        d,
        constView(macgrid_.u()),
        constView(macgrid_.v()),
        constView(macgrid_.w()),
        macgrid_.spacing());
    CUDA_CHECK(cudaGetLastError());
}

void CudaFluidSolver::solvePressure(float dt, int cycles) {
    // The previous step's pressure is kept as the initial guess (warm start).
    const float h = macgrid_.spacing();
    // A level with spacing ratio r = h_level / h has its top cell center r / 2 + 1 / 2 fine
    // cells below the zero-pressure ghost of the finest level.
    const auto weights = [this](float ratio) {
        const float open = 2.0f * ratio / (ratio + 1.0f);
        return BoundaryWeights{params_.openTop ? open : 0.0f, params_.openSides ? open : 0.0f};
    };
    std::vector<PoissonLevel> levels;
    levels.push_back(PoissonLevel{
        macgrid_.pressure().view(), constView(divergence_), GridView3D<float>(),
        residual_.view(), h * h * macgrid_.rho() / dt, weights(1.0f)});
    float ratio = 1.0f;
    for (MultigridLevel& level : coarse_levels_) {
        ratio *= 2.0f;
        levels.push_back(PoissonLevel{
            level.pressure.view(), constView(level.rhs), level.rhs.view(),
            level.residual.view(), 4.0f, weights(ratio)});
    }

    for (int cycle = 0; cycle < cycles; ++cycle) {
        vCycle(levels, 0);
    }
}

void CudaFluidSolver::applyPressureGradient(float dt) {
    const float inv_rho = 1.0f / macgrid_.rho();
    const float inv_h = 1.0f / macgrid_.spacing();
    const auto pressure = constView(macgrid_.pressure());
    const int open_top = params_.openTop ? 1 : 0;
    const int open_sides = params_.openSides ? 1 : 0;

    launchPressureGradientKernel<0>(
        macgrid_.u().view(), pressure, inv_rho, inv_h, dt, open_top, open_sides);
    launchPressureGradientKernel<1>(
        macgrid_.v().view(), pressure, inv_rho, inv_h, dt, open_top, open_sides);
    launchPressureGradientKernel<2>(
        macgrid_.w().view(), pressure, inv_rho, inv_h, dt, open_top, open_sides);
}

// Advects smoke density and temperature together; both live at cell centers.
void CudaFluidSolver::advectDensity(float dt) {
    const float h = macgrid_.spacing();
    const auto u = constView(macgrid_.u());
    const auto v = constView(macgrid_.v());
    const auto w = constView(macgrid_.w());
    const float3 offset = make_float3(0.5f, 0.5f, 0.5f);
    const float density_decay = std::exp(-params_.densityDissipation * dt);
    const float temperature_decay = std::exp(-params_.temperatureCooling * dt);

    if (!params_.maccormackScalars) {
        launchAdvect<2>({
            {constView(density_), constView(temperature_)},
            {density_next_.view(), temperature_next_.view()},
            {density_decay, temperature_decay}}, u, v, w, offset, h, dt);
        std::swap(density_, density_next_);
        std::swap(temperature_, temperature_next_);
        return;
    }

    launchAdvect<2>({
        {constView(density_), constView(temperature_)},
        {density_next_.view(), temperature_next_.view()},
        {1.0f, 1.0f}}, u, v, w, offset, h, dt);
    launchAdvect<2>({
        {constView(density_next_), constView(temperature_next_)},
        {density_scratch_.view(), temperature_scratch_.view()},
        {1.0f, 1.0f}}, u, v, w, offset, h, -dt);
    launchMaccormackCorrect<2>({
        {constView(density_), constView(temperature_)},
        {constView(density_next_), constView(temperature_next_)},
        {density_scratch_.view(), temperature_scratch_.view()},
        {density_decay, temperature_decay}}, u, v, w, offset, h, dt);
    std::swap(density_, density_scratch_);
    std::swap(temperature_, temperature_scratch_);
}

void CudaFluidSolver::step(float dt) {
    TIME(advectVelocity(dt));
    TIME(addSources(dt));
    TIME(addForces(dt));
    TIME(enforceBoundaryVelocity());

    TIME(computeDivergence());
    TIME(solvePressure(dt, params_.pressureCycles));
    TIME(applyPressureGradient(dt));
    // computeDivergence();

    TIME(advectDensity(dt));
    time_ += dt;
}
