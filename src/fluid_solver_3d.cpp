#include "fluid_solver_3d.h"
#include "cuda_check.h"

void FluidSolver3D::initDensity() {

}
void FluidSolver3D::initVelocity(){
    macgrid.u().fill(0.0f);
    macgrid.v().fill(0.0f);
    macgrid.w().fill(0.0f);
}

void FluidSolver3D::advectVelocity(float dt){
    const float inv_h = 1.0f / h_;

    for (int k = 0; k < nz_; ++k){
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i <= nx_; ++i) {
                const float x = i * h_;
                const float y = (j + 0.5f) * h_;
                const float z = (k + 0.5f) * h_;

                const Vec3 vel = macgrid.sampleVelocity(x, y, z);

                const float dep_x = x - dt * vel.x;
                const float dep_y = y - dt * vel.y;
                const float dep_z = z - dt * vel.z;

                const float u_i = dep_x * inv_h;
                const float u_j = dep_y * inv_h - 0.5f;
                const float u_k = dep_z * inv_h - 0.5f;

                u_next_(i, j, k) = sampleTrilinear(macgrid.u(), u_i, u_j, u_k);
            }
        }
    }

    for (int k = 0; k < nz_; ++k){
        for (int j = 0; j <= ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const float x = (i + 0.5f) * h_;
                const float y = j * h_;
                const float z = (k + 0.5f) * h_;

                const Vec3 vel = macgrid.sampleVelocity(x, y, z);

                const float dep_x = x - dt * vel.x;
                const float dep_y = y - dt * vel.y;
                const float dep_z = z - dt * vel.z;

                const float v_i = dep_x * inv_h - 0.5f;
                const float v_j = dep_y * inv_h;
                const float v_k = dep_z * inv_h - 0.5f;

                v_next_(i, j, k) = sampleTrilinear(macgrid.v(), v_i, v_j, v_k);
            }
        }
    }

    for (int k = 0; k <= nz_; ++k){
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const float x = (i + 0.5f) * h_;
                const float y = (j + 0.5f) * h_;
                const float z = k * h_;

                const Vec3 vel = macgrid.sampleVelocity(x, y, z);

                const float dep_x = x - dt * vel.x;
                const float dep_y = y - dt * vel.y;
                const float dep_z = z - dt * vel.z;

                const float w_i = dep_x * inv_h - 0.5f;
                const float w_j = dep_y * inv_h - 0.5f;
                const float w_k = dep_z * inv_h;

                w_next_(i, j, k) = sampleTrilinear(macgrid.w(), w_i, w_j, w_k);
            }
        }
    }

    std::swap(macgrid.u().data(), u_next_.data());
    std::swap(macgrid.v().data(), v_next_.data());
    std::swap(macgrid.w().data(), w_next_.data());
}

void FluidSolver3D::advectDensity(float dt){
    const float inv_h = 1.0f / h_;
    for (int k = 0; k < nz_; ++k)
    for (int j = 0; j < ny_; ++j) 
    for (int i = 0; i < nx_; ++i) {
        const float x = (i + 0.5f) * h_;
        const float y = (j + 0.5f) * h_;
        const float z = (k + 0.5f) * h_;

        const Vec3 vel = macgrid.sampleVelocity(x, y, z);
        const float dep_x = x - dt * vel.x;
        const float dep_y = y - dt * vel.y;
        const float dep_z = z - dt * vel.z;

        const float dep_i = dep_x * inv_h - 0.5f;
        const float dep_j = dep_y * inv_h - 0.5f;
        const float dep_k = dep_z * inv_h - 0.5f;
        density_next_(i, j, k) = sampleTrilinear(density, dep_i, dep_j, dep_k);
    }
    
    std::swap(density.data(), density_next_.data());
}

void FluidSolver3D::addForces(float dt){

}

void FluidSolver3D::addVelocitySource(float v0){
    const Vec3 center(0.5f*nx_, ny_, 0.5f*nz_);
    const float radius = 4.0f;

    for (int i = 0; i < nx_; ++i) {
        for (int k = 0; k < nz_; ++k) {
            float distance = pow((i + 0.5f - center.x)*(i + 0.5f - center.x) + (k + 0.5f - center.z)*(k + 0.5f - center.z), 0.5f);
            float weight = std::exp(-(distance*distance) / (2.0f * radius * radius));
            if (weight > 0.01f) {
                macgrid.v()(i, 1, k) = v0 * weight;
            }
        }
    }
}

void FluidSolver3D::addDensitySource(float rho){
    const Vec3 center(lroundf(0.5f*nx_), lroundf(ny_), lroundf(0.5f*nz_));
    const int half_width = 3;
    for (int i = center.x - half_width; i <= center.x + half_width; ++i) {
        for (int k = center.z - half_width; k <= center.z + half_width; ++k) {
            if (i >= 0 && k >= 0 && i < nx_ && k < nz_) {
                density(i, 0, k) = rho;
                density(i, 1, k) = rho;
            }
        }
    }
}

void FluidSolver3D::enforceBoundaryVelocity(){
    for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
            macgrid.u()(0, j, k) = 0.0f;
            macgrid.u()(nx_, j, k) = 0.0f;
        }
    }
    for (int i = 0; i < nx_; ++i) {
        for (int k = 0; k < nz_; ++k) {
            macgrid.v()(i, 0, k) = 0.0f;
            macgrid.v()(i, ny_, k) = 0.0f;
        }
    }
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            macgrid.w()(i, j, 0) = 0.0f;
            macgrid.w()(i, j, nz_) = 0.0f;
        }
    }
}

void FluidSolver3D::computeDivergence(){
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                divergence(i, j, k) = macgrid.divergenceAt(i, j, k);
            }
        }
    }
    
}
void FluidSolver3D::solvePressure(float dt, int iterations){
    macgrid.pressure().fill(0.0f);
    const float scale = h_ * h_ * rho_ / dt;
    float initial_max_update = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
        float max_update = 0.0f;
        const float* pressure = macgrid.pressure().data().data();
        float* pressure_next = pressure_next_.data().data();
        const int y_stride = nx_;
        const int z_stride = nx_ * ny_;
        const auto update_boundary_cell = [&](int i, int j, int k) {
            const int index = i + nx_ * (j + ny_ * k);
            const float updated_pressure = (
                pressure[index + (i > 0 ? -1 : 0)]
                + pressure[index + (i + 1 < nx_ ? 1 : 0)]
                + pressure[index + (j > 0 ? -y_stride : 0)]
                + pressure[index + (j + 1 < ny_ ? y_stride : 0)]
                + pressure[index + (k > 0 ? -z_stride : 0)]
                + pressure[index + (k + 1 < nz_ ? z_stride : 0)]
                - scale * divergence(i, j, k)) / 6.0f;
            max_update = std::max(max_update, std::abs(updated_pressure - pressure[index]));
            pressure_next[index] = updated_pressure;
        };

        for (int k = 0; k < nz_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                if (k == 0 || k == nz_ - 1 || j == 0 || j == ny_ - 1) {
                    for (int i = 0; i < nx_; ++i) {
                        update_boundary_cell(i, j, k);
                    }
                } else {
                    update_boundary_cell(0, j, k);
                    if (nx_ > 1) {
                        update_boundary_cell(nx_ - 1, j, k);
                    }
                }
            }
        }

        for (int k = 1; k < nz_ - 1; ++k) {
            for (int j = 1; j < ny_ - 1; ++j) {
                int index = 1 + nx_ * (j + ny_ * k);
                for (int i = 1; i < nx_ - 1; ++i, ++index) {
                    const float updated_pressure =
                        (pressure[index - 1] + pressure[index + 1]
                         + pressure[index - y_stride] + pressure[index + y_stride]
                         + pressure[index - z_stride] + pressure[index + z_stride]
                         - scale * divergence(i, j, k)) / 6.0f;
                    max_update = std::max(
                        max_update,
                        std::abs(updated_pressure - pressure[index]));
                    pressure_next[index] = updated_pressure;
                }
            }
        }
        std::swap(pressure_next_.data(), macgrid.pressure().data());
        if (iter == 0) {
            initial_max_update = max_update;
        }
        if (max_update <= initial_max_update * 1.0e-4f) {
            printf("Early break at: %d/100", iter);
            break;
        }
    }
}

void FluidSolver3D::applyPressureGradient(float dt){
    macgrid.applyPressureGradient(dt);
}

void FluidSolver3D::testDivergence(){
    float max_div = 0.0f;
    float sum_sqr = 0.0f;

    for (int k = 0; k < nz_; ++k){
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                float div = divergence(i, j, k);
                max_div = std::max(max_div, std::abs(div));
                sum_sqr += div * div;
            }
        }
    }

    float rms_div = std::sqrt(sum_sqr / (nx_ * ny_ * nz_));
    printf("Divergence max-%f, rms-%f\n", max_div, rms_div);
}

void FluidSolver3D::step(float dt){
    REPLACE(advectVelocity(dt));
    REPLACE(addVelocitySource(1.0f));
    REPLACE(addDensitySource(1.0f));
    REPLACE(addForces(dt));
    REPLACE(enforceBoundaryVelocity());

    REPLACE(computeDivergence());
    // testDivergence();
    REPLACE(solvePressure(dt, 100));
    REPLACE(applyPressureGradient(dt));
    // computeDivergence();
    // testDivergence();

    REPLACE(advectDensity(dt));
}