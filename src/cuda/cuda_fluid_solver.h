#pragma once

#include "fluid_math.h"
#include "grid.h"

#include <vector>

// A spherical smoke emitter. Positions and radii are in world units; velocity in world units per second.
struct SmokeEmitter {
    float3 center = make_float3(0.5f, 0.1f, 0.5f);
    float radius = 0.08f;
    float thickness = 0.0f;        // Half-height of a disk-shaped emitter; 0 for a sphere
    float3 velocity = make_float3(0.0f, 0.5f, 0.0f);
    float density = 1.0f;
    float temperature = 1.0f;
    float noiseAmplitude = 0.5f;   // Relative modulation of injected density / temperature
    float velocityJitter = 0.2f;   // Noise-driven velocity perturbation (world units per second)
    float noiseFrequency = 9.0f;   // Spatial frequency of the emitter noise (per world unit)
    float orbitRadius = 0.0f;      // Circular motion of the emitter center in the XZ plane
    float orbitSpeed = 0.0f;       // Radians per second
    float pulsePeriod = 0.0f;      // Seconds; 0 means continuous emission
    float pulseDuty = 1.0f;        // Fraction of each period during which the emitter is active
};

struct SmokeParameters {
    float buoyancyDensity = 0.08f;     // alpha: downward force per unit smoke density
    float buoyancyTemperature = 2.5f;  // beta: upward force per unit temperature
    float vorticity = 3.0f;            // Vorticity confinement strength
    float densityDissipation = 0.05f;  // Exponential decay rate of smoke density (1/s)
    float temperatureCooling = 0.6f;   // Exponential decay rate of temperature (1/s)
    float velocityDamping = 0.0f;      // Exponential decay rate of velocity (1/s)
    float3 wind = make_float3(0.0f, 0.0f, 0.0f); // Constant acceleration applied everywhere
    int pressureCycles = 2;            // Multigrid V-cycles per step (warm-started)
    // Second-order MacCormack advection with a clamping limiter. On velocity it amplifies
    // grid-scale noise together with vorticity confinement, so it is off by default there.
    bool maccormackScalars = true;
    bool maccormackVelocity = false;
    bool openTop = true;               // Let smoke leave through the top face of the domain
    bool openSides = true;             // Let smoke leave through the x and z faces
};

class CudaFluidSolver {
private:
    DeviceMacGrid3D macgrid_;

    DeviceGrid3D u_next_;
    DeviceGrid3D v_next_;
    DeviceGrid3D w_next_;
    DeviceGrid3D u_scratch_;
    DeviceGrid3D v_scratch_;
    DeviceGrid3D w_scratch_;
    DeviceGrid3D density_;
    DeviceGrid3D temperature_;
    DeviceGrid3D density_next_;
    DeviceGrid3D temperature_next_;
    DeviceGrid3D density_scratch_;
    DeviceGrid3D temperature_scratch_;
    DeviceGrid3D divergence_;
    DeviceGrid3D residual_;
    DeviceBuffer<float4> vorticity_;
    DeviceBuffer<float4> cell_force_;

    struct MultigridLevel {
        DeviceGrid3D pressure;
        DeviceGrid3D rhs;
        DeviceGrid3D residual;
    };
    std::vector<MultigridLevel> coarse_levels_;

    SmokeParameters params_;
    std::vector<SmokeEmitter> emitters_;
    float time_ = 0.0f;
public:
    CudaFluidSolver(int nx, int ny, int nz, float h, float rho);
    ~CudaFluidSolver();

    int nx() const noexcept { return macgrid_.nx(); }
    int ny() const noexcept { return macgrid_.ny(); }
    int nz() const noexcept { return macgrid_.nz(); }
    float spacing() const noexcept { return macgrid_.spacing(); }
    float time() const noexcept { return time_; }
    GridView3D<const float> densityView() const noexcept { return density_.view(); }
    GridView3D<const float> temperatureView() const noexcept { return temperature_.view(); }

    SmokeParameters& parameters() noexcept { return params_; }
    const SmokeParameters& parameters() const noexcept { return params_; }
    void setEmitters(std::vector<SmokeEmitter> emitters) { emitters_ = std::move(emitters); }

    void copyDensitySlice(int k, float* destination, size_t element_count) const;
    void copyDivergenceSlice(int k, float* destination, size_t element_count) const;

    void reset();
    void initDensity();
    void initVelocity();

    void advectVelocity(float dt);
    void addSources(float dt);
    void addForces(float dt);
    void enforceBoundaryVelocity();

    void computeDivergence();
    void solvePressure(float dt, int cycles);
    void applyPressureGradient(float dt);

    void advectDensity(float dt);

    void step(float dt);
};
