#  Constant Memory Complete Optimization Guide

Constant memory provides cached, broadcast-optimized access to read-only data across all threads. This guide covers advanced constant memory techniques, domain-specific applications, and optimization strategies for maximum performance.

**[Back to Overview&cuda_memory_hierarchy.md)** | **Previous: [Shared Memory Guide](3_shared_memory.md)** | **Next: [Unified Memory Guide](5_unified_memory.md)**

---

##  **Table of Contents**

1. [ Constant Memory Architecture](#-constant-memory-architecture)
2. [ Basic Usage and Optimization](#-basic-usage-and-optimization)
3. [ Scientific Computing Applications](#-scientific-computing-applications)
4. [ Graphics and Rendering Applications](#-graphics-and-rendering-applications)
5. [ Machine Learning and AI Applications](#-machine-learning-and-ai-applications)
6. [ Advanced Techniques](#-advanced-techniques)
7. [ Profiling and Performance Analysis](#-profiling-and-performance-analysis)

---

##  **Constant Memory Architecture**

Constant memory is a specialized 64KB memory space that provides cached, broadcast access to read-only data. It's optimized for scenarios where all threads in a warp read the same address simultaneously.

###  **Hardware Specifications**

| Architecture | Constant Memory Size | Cache Size | Bandwidth | Latency |
|-------------|---------------------|------------|-----------|---------|
| **Kepler** | 64 KB | 8 KB L1 | ~500 GB/s | 400-600 cycles |
| **Maxwell** | 64 KB | 8 KB L1 | ~650 GB/s | 300-400 cycles |
| **Pascal** | 64 KB | 8 KB L1 | ~900 GB/s | 200-300 cycles |
| **Volta/Turing** | 64 KB | 8 KB L1 | ~1200 GB/s | 150-200 cycles |
| **Ampere** | 64 KB | 8 KB L1 | ~1500 GB/s | 100-150 cycles |
| **Ada/Hopper** | 64 KB | 8 KB L1 | ~2000 GB/s | 80-120 cycles |

###  **Broadcast Behavior Visualization**
```
OPTIMAL: All threads read same address
Warp:     T0 T1 T2 T3 T4 T5 T6 T7   ... T31
Address:  [42] [42] [42] [42] [42] [42] [42] [42] ... [42]
Result:   |-------------- Single broadcast ---------------|
           1 memory transaction serves entire warp

SUBOPTIMAL: Different addresses per thread
Warp:     T0 T1 T2 T3 T4 T5 T6 T7   ... T31
Address:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  ... [31]
Result:   |--1--||--2--||--3--||--4--||--5--||--6--|
           Multiple transactions, no broadcast benefit
```

###  **Memory Hierarchy Integration**
```
CPU Host Memory
    ↓ cudaMemcpyToSymbol()
Constant Memory (64KB)
    ↓ Cached access
L1 Constant Cache (8KB)
    ↓ Broadcast to warp
SM Registers/Shared Memory
```

---

##  **Basic Usage and Optimization**

###  **Declaration and Initialization**

```cpp
// Declare constant memory (must be at global scope)
__constant__ float coefficients[1024];      // Filter coefficients
__constant__ float3 camera_params[16];      // Camera parameters
__constant__ int lookup_table[256];         // Lookup table
__constant__ float transformation_matrix[16]; // 4x4 matrix

// Host-side initialization
void initialize_constant_memory() {
    float host_coeffs[1024];
    // ... initialize host_coeffs ...

    // Copy to constant memory
    cudaMemcpyToSymbol(coefficients, host_coeffs, sizeof(host_coeffs));

    // Alternative: Direct initialization
    float matrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    cudaMemcpyToSymbol(transformation_matrix, matrix, sizeof(matrix));
}
```

###  **Optimal Usage Patterns**

#### **Pattern 1: Broadcast Reading (Best Performance)**
```cpp
__constant__ float filter_kernel[25];  // 5x5 convolution kernel

__global__ void convolution_optimal(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;

        // All threads read same kernel coefficients simultaneously
        for (int ky = 0; ky < 5; ++ky) {
            for (int kx = 0; kx < 5; ++kx) {
                int ix = x + kx - 2;
                int iy = y + ky - 2;

                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    //  All threads in warp read filter_kernel[ky*5+kx] together
                    sum += input[iy * width + ix] * filter_kernel[ky * 5 + kx];
                }
            }
        }

        output[y * width + x] = sum;
    }
}
```

#### **Pattern 2: Indexed Access (Good Performance)**
```cpp
__constant__ float material_properties[64][8];  // 64 materials, 8 properties each

__global__ void material_shading(int* material_ids, float3* positions,
                                float3* normals, float3* colors, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int material_id = material_ids[idx];

        // Threads may read different materials, but access is cached
        float ambient = material_properties[material_id][0];
        float diffuse = material_properties[material_id][1];
        float specular = material_properties[material_id][2];
        float shininess = material_properties[material_id][3];

        // Compute shading using constant material properties
        float3 color = compute_phong_shading(positions[idx], normals[idx],
                                           ambient, diffuse, specular, shininess);
        colors[idx] = color;
    }
}
```

###  **Suboptimal Usage Patterns**

#### **Anti-Pattern 1: Sequential Access**
```cpp
//  BAD: Sequential access doesn't utilize broadcast
__constant__ float large_array[16384];

__global__ void bad_sequential_access(float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Each thread reads different address - no broadcast benefit
        output[idx] = large_array[idx % 16384];  // Poor utilization
    }
}
```

#### **Anti-Pattern 2: Frequent Updates**
```cpp
//  BAD: Updating constant memory frequently
for (int frame = 0; frame < num_frames; ++frame) {
    // Don't do this - constant memory should be truly constant
    cudaMemcpyToSymbol(frame_params, &current_frame_data, sizeof(frame_params));
    kernel<<<grid, block>>>();  // Expensive copy every frame
}
```

---

##  **Scientific Computing Applications**

###  **Computational Chemistry: Molecular Dynamics**

```cpp
// Physical constants and force field parameters
__constant__ float atomic_masses[118];        // Periodic table masses
__constant__ float lennard_jones_params[118][118][2]; // LJ epsilon, sigma
__constant__ float bond_force_constants[100][2];      // k, r0 for bond types
__constant__ float angle_force_constants[200][2];     // k, theta0 for angles
__constant__ float coulomb_constant = 8.9875517873681764e9f;

__global__ void compute_molecular_forces(float4* positions, float4* forces,
                                       int* atom_types, int* bond_list,
                                       int* angle_list, int N_atoms, int N_bonds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_atoms) {
        float3 force = make_float3(0.0f, 0.0f, 0.0f);
        float4 pos_i = positions[idx];
        int type_i = atom_types[idx];

        // Non-bonded interactions (all threads use same LJ parameters)
        for (int j = 0; j < N_atoms; ++j) {
            if (i != j) {
                float4 pos_j = positions[j];
                int type_j = atom_types[j];

                float3 dr = make_float3(pos_j.x - pos_i.x,
                                       pos_j.y - pos_i.y,
                                       pos_j.z - pos_i.z);
                float r = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);

                //  Broadcast access to LJ parameters
                float epsilon = lennard_jones_params[type_i][type_j][0];
                float sigma = lennard_jones_params[type_i][type_j][1];

                // Compute LJ force
                float r_inv = 1.0f / r;
                float sr_inv = sigma * r_inv;
                float sr6 = sr_inv * sr_inv * sr_inv * sr_inv * sr_inv * sr_inv;
                float sr12 = sr6 * sr6;

                float force_magnitude = 24.0f * epsilon * r_inv * (2.0f * sr12 - sr6);

                force.x += force_magnitude * dr.x * r_inv;
                force.y += force_magnitude * dr.y * r_inv;
                force.z += force_magnitude * dr.z * r_inv;

                // Coulomb interaction
                float charge_product = pos_i.w * pos_j.w;  // charges in w component
                float coulomb_force = coulomb_constant * charge_product * r_inv * r_inv * r_inv;

                force.x += coulomb_force * dr.x;
                force.y += coulomb_force * dr.y;
                force.z += coulomb_force * dr.z;
            }
        }

        forces[idx] = make_float4(force.x, force.y, force.z, 0.0f);
    }
}

// Initialize force field parameters
void setup_force_field() {
    // Load atomic masses
    float masses[118] = {1.008f, 4.003f, 6.941f, 9.012f, /*...*/};
    cudaMemcpyToSymbol(atomic_masses, masses, sizeof(masses));

    // Load Lennard-Jones parameters (symmetric matrix)
    float lj_params[118][118][2];
    for (int i = 0; i < 118; ++i) {
        for (int j = 0; j < 118; ++j) {
            // Lorentz-Berthelot mixing rules
            lj_params[i][j][0] = sqrtf(epsilon[i] * epsilon[j]);  // epsilon
            lj_params[i][j][1] = 0.5f * (sigma[i] + sigma[j]);    // sigma
        }
    }
    cudaMemcpyToSymbol(lennard_jones_params, lj_params, sizeof(lj_params));
}
```

###  **Computational Fluid Dynamics: Lattice Boltzmann**

```cpp
// Lattice Boltzmann method constants
__constant__ float lattice_weights[19] = {
    1.0f/3.0f,                    // D3Q19 center weight
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, // face neighbors
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, // edge neighbors
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__constant__ int3 lattice_velocities[19] = {
    {0,0,0}, {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
    {1,1,0}, {-1,-1,0}, {1,-1,0}, {-1,1,0}, {1,0,1}, {-1,0,-1},
    {1,0,-1}, {-1,0,1}, {0,1,1}, {0,-1,-1}, {0,1,-1}, {0,-1,1}
};

__constant__ float fluid_viscosity = 0.1f;
__constant__ float relaxation_time = 0.6f;

__global__ void lbm_collision_streaming(float* f_in, float* f_out,
                                       float* density, float3* velocity,
                                       int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        int idx = z * nx * ny + y * nx + x;

        // Local distribution functions
        float f_local[19];
        float rho = 0.0f;
        float3 vel = make_float3(0.0f, 0.0f, 0.0f);

        // Load and compute macroscopic quantities
        for (int i = 0; i < 19; ++i) {
            f_local[i] = f_in[idx * 19 + i];
            rho += f_local[i];

            //  All threads broadcast access to lattice_velocities
            vel.x += f_local[i] * lattice_velocities[i].x;
            vel.y += f_local[i] * lattice_velocities[i].y;
            vel.z += f_local[i] * lattice_velocities[i].z;
        }

        vel.x /= rho;
        vel.y /= rho;
        vel.z /= rho;

        // BGK collision with equilibrium distribution
        for (int i = 0; i < 19; ++i) {
            //  Broadcast access to weights and velocities
            float3 ci = make_float3(lattice_velocities[i].x,
                                   lattice_velocities[i].y,
                                   lattice_velocities[i].z);
            float wi = lattice_weights[i];

            float ci_dot_u = ci.x * vel.x + ci.y * vel.y + ci.z * vel.z;
            float u_dot_u = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;

            // Equilibrium distribution
            float f_eq = wi * rho * (1.0f + 3.0f * ci_dot_u +
                                    4.5f * ci_dot_u * ci_dot_u -
                                    1.5f * u_dot_u);

            // Collision step
            f_local[i] += (f_eq - f_local[i]) / relaxation_time;

            // Streaming step (write to neighbor)
            int nx_next = x + lattice_velocities[i].x;
            int ny_next = y + lattice_velocities[i].y;
            int nz_next = z + lattice_velocities[i].z;

            if (nx_next >= 0 && nx_next < nx &&
                ny_next >= 0 && ny_next < ny &&
                nz_next >= 0 && nz_next < nz) {
                int next_idx = nz_next * nx * ny + ny_next * nx + nx_next;
                f_out[next_idx * 19 + i] = f_local[i];
            }
        }

        density[idx] = rho;
        velocity[idx] = vel;
    }
}
```

###  **Astrophysics: N-Body Simulation**

```cpp
// Fundamental constants and simulation parameters
__constant__ float gravitational_constant = 6.67430e-11f;  // m³/kg/s²
__constant__ float softening_parameter = 1e-3f;            // Softening length
__constant__ float time_step = 1e-6f;                      // Integration timestep

// Predefined mass categories for stellar objects
__constant__ float stellar_masses[10] = {
    1.989e30f,   // Solar mass
    3.978e30f,   // 2 solar masses
    5.967e30f,   // 3 solar masses
    1.989e31f,   // 10 solar masses
    3.978e31f,   // 20 solar masses
    9.945e31f,   // 50 solar masses
    1.989e32f,   // 100 solar masses
    2.384e34f,   // Neutron star (1.2 solar masses, compact)
    7.954e36f,   // Stellar black hole (4000 solar masses)
    7.954e38f    // Supermassive black hole (4M solar masses)
};

__global__ void nbody_force_calculation(float4* positions, float4* velocities,
                                       float3* forces, int* mass_categories, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float4 pos_i = positions[idx];
        float3 force = make_float3(0.0f, 0.0f, 0.0f);

        //  Broadcast access to mass from constant memory
        float mass_i = stellar_masses[mass_categories[idx]];

        // Compute gravitational forces from all other bodies
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                float4 pos_j = positions[j];

                // Vector from i to j
                float3 dr = make_float3(pos_j.x - pos_i.x,
                                       pos_j.y - pos_i.y,
                                       pos_j.z - pos_i.z);

                float r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z +
                          softening_parameter * softening_parameter;
                float r = sqrtf(r2);
                float r3 = r * r2;

                //  Broadcast access to mass_j
                float mass_j = stellar_masses[mass_categories[j]];

                // Newton's law of gravitation with softening
                float force_magnitude = gravitational_constant * mass_i * mass_j / r3;

                force.x += force_magnitude * dr.x;
                force.y += force_magnitude * dr.y;
                force.z += force_magnitude * dr.z;
            }
        }

        forces[idx] = force;
    }
}

// Leapfrog integration using constant timestep
__global__ void nbody_integration(float4* positions, float4* velocities,
                                 float3* forces, int* mass_categories, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float3 force = forces[idx];
        float4 pos = positions[idx];
        float4 vel = velocities[idx];

        //  Broadcast access to mass and timestep
        float mass = stellar_masses[mass_categories[idx]];
        float dt = time_step;

        // Update velocity (kick step)
        vel.x += (force.x / mass) * dt;
        vel.y += (force.y / mass) * dt;
        vel.z += (force.z / mass) * dt;

        // Update position (drift step)
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;

        positions[idx] = pos;
        velocities[idx] = vel;
    }
}
```

---

##  **Graphics and Rendering Applications**

###  **Ray Tracing: Material and Lighting**

```cpp
// Material properties for different surface types
__constant__ float4 material_albedo[16] = {
    {0.9f, 0.9f, 0.9f, 1.0f},    // White diffuse
    {0.8f, 0.2f, 0.2f, 1.0f},    // Red diffuse
    {0.2f, 0.8f, 0.2f, 1.0f},    // Green diffuse
    {0.2f, 0.2f, 0.8f, 1.0f},    // Blue diffuse
    {0.95f, 0.95f, 0.95f, 0.1f}, // Mirror (low roughness)
    {0.95f, 0.64f, 0.54f, 0.0f}, // Copper metal
    {0.95f, 0.93f, 0.88f, 0.0f}, // Silver metal
    {1.00f, 0.86f, 0.57f, 0.1f}, // Gold metal
    // ... more materials
};

__constant__ float material_roughness[16] = {
    0.8f, 0.8f, 0.8f, 0.8f,      // Diffuse materials
    0.05f, 0.1f, 0.05f, 0.1f,    // Metals
    // ... corresponding roughness values
};

__constant__ float material_metallic[16] = {
    0.0f, 0.0f, 0.0f, 0.0f,      // Non-metals
    1.0f, 1.0f, 1.0f, 1.0f,      // Pure metals
    // ... metallic factors
};

// Environment lighting (spherical harmonics coefficients)
__constant__ float3 sh_coefficients[9] = {
    {0.79f, 0.44f, 0.54f},        // L0,0
    {0.39f, 0.35f, 0.60f},        // L1,-1
    {-0.34f, -0.18f, -0.27f},     // L1,0
    {-0.29f, -0.06f, 0.01f},      // L1,1
    {-0.11f, -0.05f, -0.12f},     // L2,-2
    {-0.26f, -0.22f, -0.47f},     // L2,-1
    {-0.16f, -0.09f, -0.15f},     // L2,0
    {0.56f, 0.21f, 0.14f},        // L2,1
    {0.21f, -0.05f, -0.30f}       // L2,2
};

__device__ float3 sample_environment_lighting(float3 normal) {
    // Spherical harmonics evaluation
    float3 irradiance = sh_coefficients[0];  //  Broadcast access

    // First order (linear)
    irradiance += sh_coefficients[1] * normal.y;
    irradiance += sh_coefficients[2] * normal.z;
    irradiance += sh_coefficients[3] * normal.x;

    // Second order (quadratic)
    irradiance += sh_coefficients[4] * (normal.x * normal.y);
    irradiance += sh_coefficients[5] * (normal.y * normal.z);
    irradiance += sh_coefficients[6] * (3.0f * normal.z * normal.z - 1.0f);
    irradiance += sh_coefficients[7] * (normal.x * normal.z);
    irradiance += sh_coefficients[8] * (normal.x * normal.x - normal.y * normal.y);

    return fmaxf(irradiance, make_float3(0.0f, 0.0f, 0.0f));
}

__global__ void ray_trace_materials(float3* ray_origins, float3* ray_directions,
                                   float3* hit_points, float3* hit_normals,
                                   int* material_ids, float3* output_colors, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int mat_id = material_ids[idx];
        float3 hit_point = hit_points[idx];
        float3 normal = hit_normals[idx];
        float3 view_dir = normalize(ray_origins[idx] - hit_point);

        //  Broadcast access to material properties
        float4 albedo = material_albedo[mat_id];
        float roughness = material_roughness[mat_id];
        float metallic = material_metallic[mat_id];

        // PBR material evaluation
        float3 base_color = make_float3(albedo.x, albedo.y, albedo.z);
        float3 f0 = lerp(make_float3(0.04f, 0.04f, 0.04f), base_color, metallic);

        // Environment lighting
        float3 irradiance = sample_environment_lighting(normal);

        // Lambertian diffuse
        float3 diffuse = base_color * irradiance * (1.0f - metallic);

        // Specular reflection (simplified)
        float3 reflection_dir = reflect(-view_dir, normal);
        float3 specular_env = sample_environment_lighting(reflection_dir);
        float3 specular = f0 * specular_env;

        output_colors[idx] = diffuse + specular;
    }
}
```

###  **Real-Time Graphics: Vertex Transformation**

```cpp
// Transformation matrices (updated per frame)
__constant__ float4x4 model_matrix;
__constant__ float4x4 view_matrix;
__constant__ float4x4 projection_matrix;
__constant__ float4x4 mvp_matrix;          // Pre-multiplied for efficiency
__constant__ float4x4 normal_matrix;       // For normal transformation

// Lighting parameters
__constant__ float3 light_positions[8];    // Up to 8 lights
__constant__ float3 light_colors[8];
__constant__ float light_intensities[8];
__constant__ int active_light_count;

// Camera parameters
__constant__ float3 camera_position;
__constant__ float3 camera_forward;
__constant__ float camera_fov;
__constant__ float camera_near_plane;
__constant__ float camera_far_plane;

__global__ void vertex_transform_lighting(float3* positions, float3* normals,
                                        float2* texcoords, float3* colors,
                                        float4* output_positions, float3* output_normals,
                                        float3* output_colors, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float3 pos = positions[idx];
        float3 normal = normals[idx];

        //  All threads broadcast access to transformation matrices
        // Transform position to clip space
        float4 world_pos = mul(model_matrix, make_float4(pos.x, pos.y, pos.z, 1.0f));
        float4 clip_pos = mul(mvp_matrix, make_float4(pos.x, pos.y, pos.z, 1.0f));

        // Transform normal to world space
        float3 world_normal = normalize(mul_vec3(normal_matrix, normal));

        // Lighting calculation
        float3 total_light = make_float3(0.0f, 0.0f, 0.0f);

        for (int i = 0; i < active_light_count; ++i) {
            //  Broadcast access to light parameters
            float3 light_pos = light_positions[i];
            float3 light_color = light_colors[i];
            float light_intensity = light_intensities[i];

            float3 light_dir = normalize(light_pos - make_float3(world_pos.x, world_pos.y, world_pos.z));
            float distance = length(light_pos - make_float3(world_pos.x, world_pos.y, world_pos.z));

            // Lambertian shading
            float ndotl = fmaxf(0.0f, dot(world_normal, light_dir));
            float attenuation = 1.0f / (1.0f + 0.01f * distance + 0.001f * distance * distance);

            total_light += light_color * light_intensity * ndotl * attenuation;
        }

        output_positions[idx] = clip_pos;
        output_normals[idx] = world_normal;
        output_colors[idx] = colors[idx] * total_light;
    }
}

// Update transformation matrices (called per frame)
void update_graphics_constants(const Matrix4x4& model, const Matrix4x4& view,
                              const Matrix4x4& projection, const Camera& camera,
                              const std::vector<Light>& lights) {
    // Compute combined matrix for efficiency
    Matrix4x4 mvp = projection * view * model;
    Matrix4x4 normal = transpose(inverse(model));

    cudaMemcpyToSymbol(mvp_matrix, &mvp, sizeof(Matrix4x4));
    cudaMemcpyToSymbol(normal_matrix, &normal, sizeof(Matrix4x4));

    // Update lighting
    float3 light_pos[8], light_col[8];
    float light_int[8];

    int count = min(8, (int)lights.size());
    for (int i = 0; i < count; ++i) {
        light_pos[i] = lights[i].position;
        light_col[i] = lights[i].color;
        light_int[i] = lights[i].intensity;
    }

    cudaMemcpyToSymbol(light_positions, light_pos, sizeof(float3) * count);
    cudaMemcpyToSymbol(light_colors, light_col, sizeof(float3) * count);
    cudaMemcpyToSymbol(light_intensities, light_int, sizeof(float) * count);
    cudaMemcpyToSymbol(active_light_count, &count, sizeof(int));
}
```

---

##  **Machine Learning and AI Applications**

###  **Neural Network: Activation Functions and Weights**

```cpp
// Pre-computed activation function parameters
__constant__ float relu_alpha = 0.01f;           // Leaky ReLU negative slope
__constant__ float sigmoid_scale = 1.0f;         // Sigmoid scaling factor
__constant__ float tanh_scale = 2.0f / 3.0f;     // Scaled tanh
__constant__ float gelu_sqrt_2_pi = 0.7978845608f; // √(2/π)

// Batch normalization parameters (per layer)
__constant__ float bn_gamma[512];     // Scale parameters
__constant__ float bn_beta[512];      // Shift parameters
__constant__ float bn_mean[512];      // Running mean
__constant__ float bn_var[512];       // Running variance
__constant__ float bn_epsilon = 1e-5f;

// Dropout parameters
__constant__ float dropout_rates[16]; // Per-layer dropout rates
__constant__ bool training_mode;      // Training vs inference

__device__ float gelu_activation(float x) {
    // GELU: x * Φ(x) where Φ is standard normal CDF
    // Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    float x3 = x * x * x;
    float inner = gelu_sqrt_2_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void neural_network_forward_pass(float* input, float* output,
                                           float* weights, int* layer_sizes,
                                           int num_layers, int batch_size) {
    int batch_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;

    if (batch_idx < batch_size) {
        // Process each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            int input_size = layer_sizes[layer];
            int output_size = layer_sizes[layer + 1];

            if (neuron_idx < output_size) {
                float sum = 0.0f;

                // Matrix multiplication: output = input * weights + bias
                for (int i = 0; i < input_size; ++i) {
                    sum += input[batch_idx * input_size + i] *
                           weights[layer * input_size * output_size + i * output_size + neuron_idx];
                }

                //  Broadcast access to batch norm parameters
                float gamma = bn_gamma[layer * output_size + neuron_idx];
                float beta = bn_beta[layer * output_size + neuron_idx];
                float mean = bn_mean[layer * output_size + neuron_idx];
                float variance = bn_var[layer * output_size + neuron_idx];

                // Batch normalization
                float normalized = (sum - mean) * rsqrtf(variance + bn_epsilon);
                float bn_output = gamma * normalized + beta;

                // Activation function (GELU)
                float activated = gelu_activation(bn_output);

                // Dropout (during training)
                if (training_mode) {
                    float dropout_rate = dropout_rates[layer];
                    // Simplified dropout (would need proper random number generation)
                    if (hash_function(batch_idx, neuron_idx, layer) < dropout_rate) {
                        activated = 0.0f;
                    } else {
                        activated /= (1.0f - dropout_rate);  // Scale for inference
                    }
                }

                output[batch_idx * output_size + neuron_idx] = activated;
            }
            __syncthreads();
        }
    }
}
```

###  **Natural Language Processing: Transformer Attention**

```cpp
// Transformer model parameters
__constant__ int model_dim = 512;
__constant__ int num_heads = 8;
__constant__ int head_dim = 64;  // model_dim / num_heads
__constant__ int max_sequence_length = 2048;
__constant__ float attention_scale;  // 1 / sqrt(head_dim)

// Position encoding (sinusoidal)
__constant__ float positional_encoding[2048][512];

// Layer normalization parameters
__constant__ float ln_gamma[512];
__constant__ float ln_beta[512];
__constant__ float ln_epsilon = 1e-6f;

// Multi-head attention weights (query, key, value projections)
__constant__ float attention_weights_q[512][512];
__constant__ float attention_weights_k[512][512];
__constant__ float attention_weights_v[512][512];
__constant__ float attention_weights_out[512][512];

__global__ void transformer_attention(float* input_embeddings, float* output,
                                    int* attention_mask, int batch_size,
                                    int sequence_length) {
    int batch_idx = blockIdx.x;
    int seq_pos = blockIdx.y;
    int thread_idx = threadIdx.x;

    if (batch_idx < batch_size && seq_pos < sequence_length && thread_idx < model_dim) {
        // Add positional encoding
        float input_with_pos = input_embeddings[batch_idx * sequence_length * model_dim +
                                              seq_pos * model_dim + thread_idx] +
                              positional_encoding[seq_pos][thread_idx];

        // Layer normalization
        __shared__ float layer_sum, layer_mean, layer_var;

        if (thread_idx == 0) {
            layer_sum = 0.0f;
            for (int i = 0; i < model_dim; ++i) {
                layer_sum += input_with_pos;  // Simplified - should sum across all dims
            }
            layer_mean = layer_sum / model_dim;
        }
        __syncthreads();

        float centered = input_with_pos - layer_mean;

        if (thread_idx == 0) {
            layer_var = 0.0f;
            for (int i = 0; i < model_dim; ++i) {
                layer_var += centered * centered;  // Simplified
            }
            layer_var /= model_dim;
        }
        __syncthreads();

        //  Broadcast access to layer norm parameters
        float gamma = ln_gamma[thread_idx];
        float beta = ln_beta[thread_idx];
        float normalized = gamma * centered * rsqrtf(layer_var + ln_epsilon) + beta;

        // Multi-head attention
        __shared__ float shared_q[512], shared_k[512], shared_v[512];

        // Query, Key, Value projections
        if (thread_idx < model_dim) {
            float q = 0.0f, k = 0.0f, v = 0.0f;

            for (int i = 0; i < model_dim; ++i) {
                //  Broadcast access to attention weights
                q += normalized * attention_weights_q[i][thread_idx];
                k += normalized * attention_weights_k[i][thread_idx];
                v += normalized * attention_weights_v[i][thread_idx];
            }

            shared_q[thread_idx] = q;
            shared_k[thread_idx] = k;
            shared_v[thread_idx] = v;
        }
        __syncthreads();

        // Scaled dot-product attention (simplified)
        if (thread_idx < head_dim) {
            int head_idx = thread_idx / head_dim;
            int dim_idx = thread_idx % head_dim;

            float attention_sum = 0.0f;

            // Compute attention weights for current position
            for (int other_pos = 0; other_pos < sequence_length; ++other_pos) {
                if (attention_mask[seq_pos * sequence_length + other_pos]) {
                    float score = 0.0f;

                    // Dot product of query and key
                    for (int d = 0; d < head_dim; ++d) {
                        score += shared_q[head_idx * head_dim + d] *
                                shared_k[head_idx * head_dim + d];
                    }

                    //  Broadcast access to attention scale
                    score *= attention_scale;
                    attention_sum += expf(score) * shared_v[head_idx * head_dim + dim_idx];
                }
            }

            output[batch_idx * sequence_length * model_dim +
                  seq_pos * model_dim + thread_idx] = attention_sum;
        }
    }
}
```

###  **Computer Vision: Convolutional Neural Networks**

```cpp
// CNN Layer parameters
__constant__ float conv_kernels[64][3][3]; // 64 3x3 kernels for first layer
__constant__ float conv_biases[64];
__constant__ float conv_scales[64];        // For batch normalization

// Pooling parameters
__constant__ int pool_size = 2;
__constant__ int pool_stride = 2;

// Activation parameters
__constant__ float leaky_relu_slope = 0.01f;

__global__ void cnn_convolution_layer(float* input, float* output,
                                     int input_width, int input_height, int input_channels,
                                     int output_channels, int batch_size) {
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;
    int out_channel = threadIdx.x;

    if (batch_idx < batch_size && out_y < input_height - 2 &&
        out_x < input_width - 2 && out_channel < output_channels) {

        float sum = 0.0f;

        // Convolution operation
        for (int in_ch = 0; in_ch < input_channels; ++in_ch) {
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;

                    float input_val = input[batch_idx * input_channels * input_height * input_width +
                                          in_ch * input_height * input_width +
                                          in_y * input_width + in_x];

                    //  Broadcast access to kernel weights
                    float kernel_val = conv_kernels[out_channel][ky][kx];

                    sum += input_val * kernel_val;
                }
            }
        }

        //  Add bias (broadcast)
        sum += conv_biases[out_channel];

        // Batch normalization (simplified)
        sum *= conv_scales[out_channel];

        // Leaky ReLU activation
        float activated = (sum > 0.0f) ? sum : leaky_relu_slope * sum;

        output[batch_idx * output_channels * (input_height-2) * (input_width-2) +
               out_channel * (input_height-2) * (input_width-2) +
               out_y * (input_width-2) + out_x] = activated;
    }
}
```

---

##  **Advanced Techniques**

###  **1. Dynamic Constant Memory Loading**

```cpp
template<typename T>
class ConstantMemoryManager {
private:
    static constexpr size_t MAX_CONSTANT_SIZE = 64 * 1024;  // 64KB limit
    size_t current_offset = 0;

public:
    template<size_t N>
    cudaError_t load_array(T (&symbol)[N], const T* host_data) {
        size_t required_size = N * sizeof(T);

        if (current_offset + required_size > MAX_CONSTANT_SIZE) {
            return cudaErrorMemoryAllocation;  // Not enough constant memory
        }

        cudaError_t error = cudaMemcpyToSymbol(symbol, host_data, required_size);
        if (error == cudaSuccess) {
            current_offset += required_size;
        }

        return error;
    }

    void reset() { current_offset = 0; }
    size_t get_remaining_space() const { return MAX_CONSTANT_SIZE - current_offset; }
};

// Usage example
ConstantMemoryManager<float> const_manager;

void load_physics_constants() {
    float masses[100] = {/* ... */};
    float charges[100] = {/* ... */};
    float3 positions[50] = {/* ... */};

    const_manager.load_array(particle_masses, masses);
    const_manager.load_array(particle_charges, charges);
    const_manager.load_array(fixed_positions, positions);

    printf("Remaining constant memory: %zu bytes\n", const_manager.get_remaining_space());
}
```

###  **2. Constant Memory with Templates**

```cpp
template<int MAX_MATERIALS>
struct MaterialSystem {
    __constant__ static float4 albedo[MAX_MATERIALS];
    __constant__ static float roughness[MAX_MATERIALS];
    __constant__ static float metallic[MAX_MATERIALS];
    __constant__ static float emission[MAX_MATERIALS];

    static void initialize(const std::vector<Material>& materials) {
        static_assert(MAX_MATERIALS <= 256, "Too many materials for constant memory");

        float4 host_albedo[MAX_MATERIALS];
        float host_roughness[MAX_MATERIALS];
        float host_metallic[MAX_MATERIALS];
        float host_emission[MAX_MATERIALS];

        for (size_t i = 0; i < min(materials.size(), (size_t)MAX_MATERIALS); ++i) {
            host_albedo[i] = materials[i].albedo;
            host_roughness[i] = materials[i].roughness;
            host_metallic[i] = materials[i].metallic;
            host_emission[i] = materials[i].emission;
        }

        cudaMemcpyToSymbol(albedo, host_albedo, sizeof(host_albedo));
        cudaMemcpyToSymbol(roughness, host_roughness, sizeof(host_roughness));
        cudaMemcpyToSymbol(metallic, host_metallic, sizeof(host_metallic));
        cudaMemcpyToSymbol(emission, host_emission, sizeof(host_emission));
    }
};

// Instantiate for specific use case
using StandardMaterials = MaterialSystem<64>;   // 64 materials
using ExtendedMaterials = MaterialSystem<256>;  // 256 materials (uses more constant memory)
```

###  **3. Constant Memory Pooling**

```cpp
// Shared constant memory pool for multiple systems
class ConstantMemoryPool {
private:
    struct Allocation {
        size_t offset;
        size_t size;
        bool active;
    };

    std::vector<Allocation> allocations;
    size_t total_used = 0;
    static constexpr size_t POOL_SIZE = 60 * 1024;  // Leave some headroom

public:
    template<typename T>
    bool allocate_space(size_t count, size_t& out_offset) {
        size_t required = count * sizeof(T);

        if (total_used + required > POOL_SIZE) {
            return false;
        }

        out_offset = total_used;
        allocations.push_back({total_used, required, true});
        total_used += required;

        return true;
    }

    void deallocate(size_t offset) {
        for (auto& alloc : allocations) {
            if (alloc.offset == offset) {
                alloc.active = false;
                // Note: Actual deallocation would require compaction
                break;
            }
        }
    }

    void print_usage() {
        printf("Constant memory pool usage: %.1f%% (%zu / %zu bytes)\n",
               100.0f * total_used / POOL_SIZE, total_used, POOL_SIZE);
    }
};
```

###  **4. Compile-Time Constant Memory Optimization**

```cpp
template<typename ConstantData>
struct ConstantMemoryOptimizer {
    // Check if data fits in constant memory at compile time
    static_assert(sizeof(ConstantData) <= 65536,
                  "Data too large for constant memory");

    // Suggest alternative if too large
    using AlternativeStorage = typename std::conditional<
        sizeof(ConstantData) > 65536,
        GlobalMemoryStorage<ConstantData>,
        ConstantMemoryStorage<ConstantData>
    >::type;

    // Automatically choose best access pattern
    static constexpr bool USE_BROADCAST =
        ConstantData::has_uniform_access_pattern();

    static constexpr bool USE_CACHING =
        !USE_BROADCAST && ConstantData::has_locality();
};

// Usage
struct PhysicsConstants {
    float gravitational_constant;
    float speed_of_light;
    float planck_constant;
    // ...

    static constexpr bool has_uniform_access_pattern() { return true; }
    static constexpr bool has_locality() { return false; }
};

using OptimalPhysicsStorage = ConstantMemoryOptimizer<PhysicsConstants>::AlternativeStorage;
```

---

##  **Profiling and Performance Analysis**

###  **Nsight Compute Metrics for Constant Memory**

```bash
# Check constant memory cache hit rate
ncu --metrics l1tex__t_sector_hit_rate,l1tex__t_sector_miss_rate ./app

# Analyze constant memory throughput
ncu --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed ./app

# Check for broadcast efficiency
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_lg_op_ld.sum ./app

# Comprehensive constant memory analysis
ncu --set full --section MemoryWorkloadAnalysis_ConstantMemory ./app
```

###  **Performance Validation**

```cpp
// Benchmark constant memory vs global memory access
template<typename T, size_t N>
float benchmark_constant_vs_global(T (&constant_array)[N], T* global_array, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(256);
    dim3 grid((1024 + block.x - 1) / block.x);

    // Benchmark constant memory access
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        constant_memory_kernel<<<grid, block>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float constant_time;
    cudaEventElapsedTime(&constant_time, start, stop);

    // Benchmark global memory access
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        global_memory_kernel<<<grid, block>>>(global_array);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float global_time;
    cudaEventElapsedTime(&global_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Constant memory: %.3f ms\n", constant_time / iterations);
    printf("Global memory: %.3f ms\n", global_time / iterations);
    printf("Speedup: %.2fx\n", global_time / constant_time);

    return global_time / constant_time;
}
```

###  **Cache Performance Analysis**

```cpp
__global__ void constant_cache_test(float* output, int access_pattern) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0.0f;

    switch (access_pattern) {
        case 0: // Broadcast - optimal
            result = filter_kernel[0];  // All threads read same value
            break;

        case 1: // Sequential - suboptimal
            result = filter_kernel[idx % 25];  // Different values per thread
            break;

        case 2: // Random - worst
            result = filter_kernel[(idx * 17 + 23) % 25];  // Pseudo-random access
            break;
    }

    output[idx] = result;
}

void analyze_cache_performance() {
    float* d_output;
    cudaMalloc(&d_output, 1024 * sizeof(float));

    dim3 block(256);
    dim3 grid(4);

    printf("Constant memory cache performance analysis:\n");

    for (int pattern = 0; pattern < 3; ++pattern) {
        float time = benchmark_kernel_time([&]() {
            constant_cache_test<<<grid, block>>>(d_output, pattern);
        });

        const char* pattern_names[] = {"Broadcast", "Sequential", "Random"};
        printf("  %s access: %.3f ms\n", pattern_names[pattern], time);
    }

    cudaFree(d_output);
}
```

---

##  **Key Takeaways**

1. ** Design for Broadcast**: Constant memory shines when all threads read the same address
2. ** 64KB Limit**: Plan your data layout carefully - constant memory is limited
3. ** Cache Efficiently**: Even non-broadcast access benefits from L1 constant cache
4. ** Perfect for Parameters**: Model parameters, physical constants, and lookup tables are ideal
5. ** Profile Access Patterns**: Use Nsight Compute to validate cache hit rates

##  **Related Guides**

- **Next Step**: [Unified Memory Complete Guide](5_unified_memory.md) - Simplify memory management
- **Previous**: [Shared Memory Complete Guide](3_shared_memory.md) - Fast on-chip memory
- **Debugging**: [Memory Debugging Toolkit](6_memory_debugging.md) - Troubleshoot memory issues
- **Overview**: [Memory Hierarchy Overview&cuda_memory_hierarchy.md) - Quick reference and navigation

---

** Pro Tip**: Use constant memory for data that doesn't change during kernel execution and is accessed by multiple threads. The broadcast capability makes it incredibly efficient for shared parameters and lookup tables!
