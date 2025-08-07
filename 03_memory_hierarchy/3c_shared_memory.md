#  Shared Memory Complete Optimization Guide

Shared memory is the key to unlocking maximum GPU performance. This guide covers advanced tiling techniques, bank conflict avoidance, and shared memory optimization patterns for high-performance CUDA kernels.

**[Back to Overview](3_cuda_memory_hierarchy.md)** | **Previous: [Global Memory Guide](3b_global_memory.md)** | **Next: [Constant Memory Guide](3d_constant_memory.md)**

---

##  **Table of Contents**

1. [ Shared Memory Architecture](#-shared-memory-architecture)
2. [ Matrix Multiplication Tiling Mastery](#-matrix-multiplication-tiling-mastery)
3. [ Bank Conflict Deep Dive](#-bank-conflict-deep-dive)
4. [ Advanced Tiling Patterns](#-advanced-tiling-patterns)
5. [ Performance Optimization Techniques](#-performance-optimization-techniques)
6. [ Profiling and Debugging](#-profiling-and-debugging)
7. [ Real-World Applications](#-real-world-applications)

---

##  **Shared Memory Architecture**

Shared memory is on-chip memory shared by all threads in a block. It's approximately 100x faster than global memory but requires careful management to avoid conflicts and maximize utilization.

###  **Hardware Specifications**

| Architecture | Shared Memory/SM | Banks | Bank Width | Max Bandwidth |
|-------------|------------------|-------|------------|---------------|
| **Kepler** | 48/64/96 KB | 32 | 4 bytes | ~1.5 TB/s |
| **Maxwell** | 48/64/96 KB | 32 | 4 bytes | ~2.0 TB/s |
| **Pascal** | 48/64/96 KB | 32 | 4 bytes | ~3.0 TB/s |
| **Volta/Turing** | 96 KB | 32 | 4 bytes | ~5.0 TB/s |
| **Ampere** | 100 KB | 32 | 4 bytes | ~6.0 TB/s |
| **Ada/Hopper** | 100 KB | 32 | 4 bytes | ~8.0 TB/s |

###  **Memory Bank Layout Visualization**
```
Shared Memory Banks (32 banks, 4-byte width):
Bank:    0 1    2 3    4   ...   31 0    1 2    3   ...
Address: 0 4    8 12 16   ...  124 128 132 136 140  ...
         |----| |----| |----| |----| |----| ... |----| |----| |----| |----|

Sequential Access (No Conflicts):
Thread:  T0 T1 T2 T3 T4   ...  T31 T0 T1 T2 T3   ...
Access:  [0]  [4]  [8] [12] [16]  ... [124] [128][132][136][140] ...
Result:   All threads access different banks → Conflict-free
```

---

##  **Matrix Multiplication Tiling Mastery**

Matrix multiplication is the classic example for shared memory optimization. Let's build from basic to advanced implementations.

###  **Performance Progression Overview**
| Implementation | GFLOPS | Efficiency | Memory Pattern |
|---------------|--------|------------|---------------|
| **Naive Global** | 45 | 2% | Poor coalescing |
| **Basic Tiling** | 890 | 38% | Good coalescing |
| **Bank Conflict-Free** | 1,200 | 51% | Optimal access |
| **Double Buffering** | 1,450 | 62% | Overlap compute/memory |
| **Tensor Core Optimized** | 2,800 | 85% | Hardware acceleration |

###  **Level 1: Basic Tiling Implementation**

```cpp
#define TILE_SIZE 16

__global__ void matmul_tiled_basic(float* A, float* B, float* C, int N) {
    // Shared memory tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile from A
        if (row < N && tile * TILE_SIZE + threadIdx.x < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B
        if (col < N && tile * TILE_SIZE + threadIdx.y < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Store result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

###  **Level 2: Bank Conflict-Free Version**

```cpp
#define TILE_SIZE 16
#define PADDING 1  // Avoid bank conflicts

__global__ void matmul_no_conflicts(float* A, float* B, float* C, int N) {
    // Add padding to avoid bank conflicts
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + PADDING];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + PADDING];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Coalesced loading with bounds checking
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        tile_A[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        tile_B[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Unrolled computation for better performance
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

###  **Level 3: Double Buffering for Maximum Performance**

```cpp
#define TILE_SIZE 16
#define PADDING 1

__global__ void matmul_double_buffered(float* A, float* B, float* C, int N) {
    // Double buffering: 2 sets of tiles
    __shared__ float tile_A[2][TILE_SIZE][TILE_SIZE + PADDING];
    __shared__ float tile_B[2][TILE_SIZE][TILE_SIZE + PADDING];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Pre-load first tile
    int current_buffer = 0;
    if (num_tiles > 0) {
        tile_A[current_buffer][ty][tx] = (row < N && tx < N) ? A[row * N + tx] : 0.0f;
        tile_B[current_buffer][ty][tx] = (ty < N && col < N) ? B[ty * N + col] : 0.0f;
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        __syncthreads();

        // Start loading next tile while computing current
        int next_buffer = 1 - current_buffer;
        if (tile + 1 < num_tiles) {
            int next_tile_offset = (tile + 1) * TILE_SIZE;
            int a_col = next_tile_offset + tx;
            int b_row = next_tile_offset + ty;

            tile_A[next_buffer][ty][tx] = (row < N && a_col < N) ?
                A[row * N + a_col] : 0.0f;
            tile_B[next_buffer][ty][tx] = (b_row < N && col < N) ?
                B[b_row * N + col] : 0.0f;
        }

        // Compute using current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[current_buffer][ty][k] * tile_B[current_buffer][k][tx];
        }

        current_buffer = next_buffer;
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

##  **Bank Conflict Deep Dive**

Bank conflicts occur when multiple threads in a warp try to access the same memory bank simultaneously. Understanding and avoiding them is crucial for shared memory performance.

###  **Bank Conflict Visualization**

####  **Conflict-Free Access**
```cpp
__shared__ float data[32];

// All threads access different banks
int tid = threadIdx.x;
float val = data[tid];  // Thread 0→Bank 0, Thread 1→Bank 1, etc.
```

```
Threads:  T0 T1 T2 T3 T4 T5 T6 T7  ... T31
Banks:    B0 B1 B2 B3 B4 B5 B6 B7  ... B31
Access:   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ... ↓
Result:    No conflicts - single cycle access
```

####  **2-Way Bank Conflict**
```cpp
__shared__ float data[64];

// Two threads access same bank
int tid = threadIdx.x;
float val = data[tid * 2];  // Stride 2 access pattern
```

```
Threads:  T0 T1 T2 T3 T4 T5 T6 T7  ... T15 T16 ...
Indices:  0 2   4 6   8 10 12 14  ... 30 32 ...
Banks:    B0 B2 B4 B6 B8 B10 B12 B14 ... B30 B0 ...
                                                    ↑
Result:    T0 and T16 both access Bank 0 → 2-way conflict
```

####  **N-Way Bank Conflict (Worst Case)**
```cpp
__shared__ float data[1024];

// All threads access same bank
int tid = threadIdx.x;
float val = data[0];  // All threads read index 0
```

```
Threads:  T0 T1 T2 T3 T4 T5 T6 T7  ... T31
Indices:  0 0   0 0   0 0   0 0   ... 0
Banks:    B0 B0 B0 B0 B0 B0 B0 B0  ... B0
Result:    32-way conflict → 32x slower access
```

###  **Bank Conflict Solutions**

#### **1. Padding to Eliminate Conflicts**
```cpp
// Problem: 2D array with stride causing conflicts
__shared__ float matrix[16][16];  // Bank conflicts on column access

// Solution: Add padding
__shared__ float matrix[16][17];  // Extra column eliminates conflicts

// Column access now conflict-free
for (int row = 0; row < 16; row++) {
    float val = matrix[row][threadIdx.x];  // No conflicts!
}
```

#### **2. Array Reordering**
```cpp
// Transform conflicting pattern into conflict-free pattern
__global__ void transpose_shared_optimized(float* input, float* output, int N) {
    __shared__ float tile[16][17];  // Padded for conflict-free access

    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;

    // Coalesced read, conflict-free write to shared memory
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();

    // Transpose coordinates for output
    x = blockIdx.y * 16 + threadIdx.x;
    y = blockIdx.x * 16 + threadIdx.y;

    // Conflict-free read, coalesced write to global memory
    if (x < N && y < N) {
        output[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

#### **3. Dynamic Padding Calculation**
```cpp
template<int TILE_SIZE>
__device__ constexpr int calculate_padding() {
    // Ensure no bank conflicts for square tiles
    return (32 % TILE_SIZE == 0) ? 1 : 0;
}

#define TILE_SIZE 16
#define PADDING calculate_padding<TILE_SIZE>()

__shared__ float shared_data[TILE_SIZE][TILE_SIZE + PADDING];
```

---

##  **Advanced Tiling Patterns**

###  **Rectangular Tiling**
```cpp
#define TILE_WIDTH 32
#define TILE_HEIGHT 8

__global__ void rectangular_tiling_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_HEIGHT][TILE_WIDTH + 1];
    __shared__ float tile_B[TILE_WIDTH][TILE_HEIGHT + 1];

    int row = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // Optimized for specific matrix dimensions and memory hierarchy
    for (int tile = 0; tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
        // Load rectangular tiles optimized for memory coalescing
        if (row < M && tile * TILE_WIDTH + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_WIDTH + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tile * TILE_WIDTH + threadIdx.y < K && col < N) {
            tile_B[threadIdx.x][threadIdx.y] = B[(tile * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

###  **Multi-Level Tiling**
```cpp
#define OUTER_TILE 64
#define INNER_TILE 16

__global__ void multi_level_tiling(float* A, float* B, float* C, int N) {
    // Level 1: Outer tile in shared memory
    __shared__ float outer_A[OUTER_TILE][OUTER_TILE + 1];
    __shared__ float outer_B[OUTER_TILE][OUTER_TILE + 1];

    // Level 2: Inner tiles in registers
    float inner_A[INNER_TILE];
    float inner_B[INNER_TILE];
    float results[INNER_TILE][INNER_TILE] = {0};

    int block_row = blockIdx.y * OUTER_TILE;
    int block_col = blockIdx.x * OUTER_TILE;

    for (int outer_k = 0; outer_k < N; outer_k += OUTER_TILE) {
        // Load outer tiles cooperatively
        for (int i = threadIdx.y; i < OUTER_TILE; i += blockDim.y) {
            for (int j = threadIdx.x; j < OUTER_TILE; j += blockDim.x) {
                if (block_row + i < N && outer_k + j < N) {
                    outer_A[i][j] = A[(block_row + i) * N + outer_k + j];
                } else {
                    outer_A[i][j] = 0.0f;
                }

                if (outer_k + i < N && block_col + j < N) {
                    outer_B[i][j] = B[(outer_k + i) * N + block_col + j];
                } else {
                    outer_B[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Process inner tiles
        for (int inner_k = 0; inner_k < OUTER_TILE; inner_k += INNER_TILE) {
            // Load inner tiles to registers
            #pragma unroll
            for (int i = 0; i < INNER_TILE; ++i) {
                inner_A[i] = outer_A[threadIdx.y * INNER_TILE + i][inner_k + threadIdx.x];
                inner_B[i] = outer_B[inner_k + threadIdx.y][threadIdx.x * INNER_TILE + i];
            }

            // Compute inner matrix multiplication
            #pragma unroll
            for (int i = 0; i < INNER_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < INNER_TILE; ++j) {
                    results[i][j] += inner_A[i] * inner_B[j];
                }
            }
        }
        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < INNER_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < INNER_TILE; ++j) {
            int row = block_row + threadIdx.y * INNER_TILE + i;
            int col = block_col + threadIdx.x * INNER_TILE + j;
            if (row < N && col < N) {
                C[row * N + col] = results[i][j];
            }
        }
    }
}
```

---

##  **Performance Optimization Techniques**

###  **1. Occupancy Optimization**

```cpp
// Calculate optimal shared memory usage for max occupancy
template<int BLOCK_SIZE>
constexpr int calculate_optimal_tile_size() {
    constexpr int max_shared_per_block = 48 * 1024;  // 48KB typical
    constexpr int bytes_per_element = sizeof(float);
    constexpr int elements_per_tile = BLOCK_SIZE * BLOCK_SIZE;
    constexpr int max_tiles = max_shared_per_block / (elements_per_tile * bytes_per_element);

    // Return largest power of 2 that fits
    return (max_tiles >= 4) ? 32 : (max_tiles >= 2) ? 16 : 8;
}

#define BLOCK_SIZE 16
#define OPTIMAL_TILE calculate_optimal_tile_size<BLOCK_SIZE>()
```

###  **2. Register Optimization**

```cpp
__global__ void register_optimized_kernel(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[16][17];
    __shared__ float tile_B[16][17];

    // Use registers to store frequently accessed values
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * 16 + ty;
    const int col = blockIdx.x * 16 + tx;

    // Register arrays for accumulation
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Process 4 elements per thread

    for (int tile = 0; tile < (N + 15) / 16; ++tile) {
        // Load with register reuse
        float a_reg = (row < N && tile * 16 + tx < N) ?
                     A[row * N + tile * 16 + tx] : 0.0f;
        float b_reg = (tile * 16 + ty < N && col < N) ?
                     B[(tile * 16 + ty) * N + col] : 0.0f;

        tile_A[ty][tx] = a_reg;
        tile_B[ty][tx] = b_reg;

        __syncthreads();

        // Unrolled computation with register optimization
        #pragma unroll 4
        for (int k = 0; k < 16; k += 4) {
            float b_vals[4] = {tile_B[k][tx], tile_B[k+1][tx],
                              tile_B[k+2][tx], tile_B[k+3][tx]};

            #pragma unroll 4
            for (int i = 0; i < 4; ++i) {
                sum[i] += tile_A[ty][k+i] * b_vals[i];
            }
        }

        __syncthreads();
    }

    // Store accumulated results
    if (row < N && col < N) {
        C[row * N + col] = sum[0] + sum[1] + sum[2] + sum[3];
    }
}
```

###  **3. Memory Access Pattern Optimization**

```cpp
// Optimize for specific access patterns
__global__ void pattern_optimized_convolution(float* input, float* kernel,
                                            float* output, int width, int height) {
    const int TILE_SIZE = 16;
    const int KERNEL_SIZE = 5;
    const int SHARED_SIZE = TILE_SIZE + KERNEL_SIZE - 1;

    __shared__ float shared_input[SHARED_SIZE][SHARED_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int output_x = blockIdx.x * TILE_SIZE + tx;
    int output_y = blockIdx.y * TILE_SIZE + ty;

    // Cooperative loading with optimal access pattern
    for (int i = ty; i < SHARED_SIZE; i += blockDim.y) {
        for (int j = tx; j < SHARED_SIZE; j += blockDim.x) {
            int input_x = blockIdx.x * TILE_SIZE + j - KERNEL_SIZE/2;
            int input_y = blockIdx.y * TILE_SIZE + i - KERNEL_SIZE/2;

            if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
                shared_input[i][j] = input[input_y * width + input_x];
            } else {
                shared_input[i][j] = 0.0f;  // Padding
            }
        }
    }
    __syncthreads();

    // Compute convolution using shared memory
    if (output_x < width && output_y < height) {
        float sum = 0.0f;

        #pragma unroll
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                sum += shared_input[ty + ky][tx + kx] * kernel[ky * KERNEL_SIZE + kx];
            }
        }

        output[output_y * width + output_x] = sum;
    }
}
```

---

##  **Profiling and Debugging**

###  **Essential Shared Memory Metrics**

```bash
# Check shared memory bank conflicts
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum ./app

# Check shared memory utilization
ncu --metrics shared_load_transactions_per_request,shared_store_transactions_per_request ./app

# Comprehensive shared memory analysis
ncu --set full --section SharedMemoryStatistics ./app
```

###  **Bank Conflict Detection**

```bash
# Specific bank conflict metrics
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./app

# Bank conflict ratio (should be close to 1.0 for optimal performance)
ncu --metrics l1tex__average_bank_conflicts_pipe_lsu_mem_shared_op_ld ./app
```

###  **Optimization Validation**

```cpp
// Shared memory performance testing harness
template<typename KernelFunc>
float benchmark_shared_memory_kernel(KernelFunc kernel,
                                    float* d_A, float* d_B, float* d_C,
                                    int N, int iterations = 100) {
    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Grid configuration
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Warm-up
    kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;
}

// Usage example
float time_basic = benchmark_shared_memory_kernel(matmul_tiled_basic, d_A, d_B, d_C, N);
float time_optimized = benchmark_shared_memory_kernel(matmul_no_conflicts, d_A, d_B, d_C, N);

printf("Speedup from bank conflict elimination: %.2fx\n", time_basic / time_optimized);
```

---

##  **Real-World Applications**

###  **Scientific Computing: Stencil Operations**

```cpp
// 3D heat diffusion with shared memory optimization
__global__ void heat_diffusion_3d_shared(float* u, float* u_new,
                                        int nx, int ny, int nz, float dt, float dx) {
    const int BLOCK_X = 8, BLOCK_Y = 8, BLOCK_Z = 8;
    const int SHARED_X = BLOCK_X + 2, SHARED_Y = BLOCK_Y + 2, SHARED_Z = BLOCK_Z + 2;

    __shared__ float shared_u[SHARED_Z][SHARED_Y][SHARED_X];

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int x = blockIdx.x * BLOCK_X + tx;
    int y = blockIdx.y * BLOCK_Y + ty;
    int z = blockIdx.z * BLOCK_Z + tz;

    // Cooperative loading with halo regions
    for (int sz = tz; sz < SHARED_Z; sz += BLOCK_Z) {
        for (int sy = ty; sy < SHARED_Y; sy += BLOCK_Y) {
            for (int sx = tx; sx < SHARED_X; sx += BLOCK_X) {
                int gx = blockIdx.x * BLOCK_X + sx - 1;
                int gy = blockIdx.y * BLOCK_Y + sy - 1;
                int gz = blockIdx.z * BLOCK_Z + sz - 1;

                if (gx >= 0 && gx < nx && gy >= 0 && gy < ny && gz >= 0 && gz < nz) {
                    shared_u[sz][sy][sx] = u[gz * nx * ny + gy * nx + gx];
                } else {
                    shared_u[sz][sy][sx] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    // Compute stencil
    if (x > 0 && x < nx-1 && y > 0 && y < ny-1 && z > 0 && z < nz-1) {
        float laplacian =
            shared_u[tz+1][ty+1][tx+2] + shared_u[tz+1][ty+1][tx]   +  // x direction
            shared_u[tz+1][ty+2][tx+1] + shared_u[tz+1][ty][tx+1]   +  // y direction
            shared_u[tz+2][ty+1][tx+1] + shared_u[tz][ty+1][tx+1]   -  // z direction
            6.0f * shared_u[tz+1][ty+1][tx+1];

        u_new[z * nx * ny + y * nx + x] = shared_u[tz+1][ty+1][tx+1] +
                                         dt * laplacian / (dx * dx);
    }
}
```

###  **Graphics: Fast Gaussian Blur**

```cpp
__global__ void gaussian_blur_shared(unsigned char* input, unsigned char* output,
                                   int width, int height, float* kernel, int kernel_size) {
    const int TILE_SIZE = 16;
    const int SHARED_SIZE = TILE_SIZE + kernel_size - 1;

    __shared__ float shared_data[SHARED_SIZE][SHARED_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    int half_kernel = kernel_size / 2;

    // Load data with padding
    for (int i = ty; i < SHARED_SIZE; i += blockDim.y) {
        for (int j = tx; j < SHARED_SIZE; j += blockDim.x) {
            int src_x = blockIdx.x * TILE_SIZE + j - half_kernel;
            int src_y = blockIdx.y * TILE_SIZE + i - half_kernel;

            // Clamp to image boundaries
            src_x = max(0, min(src_x, width - 1));
            src_y = max(0, min(src_y, height - 1));

            shared_data[i][j] = (float)input[src_y * width + src_x];
        }
    }
    __syncthreads();

    // Apply Gaussian filter
    if (x < width && y < height) {
        float sum = 0.0f;

        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                sum += shared_data[ty + ky][tx + kx] * kernel[ky * kernel_size + kx];
            }
        }

        output[y * width + x] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum));
    }
}
```

---

##  **Key Takeaways**

1. ** Always Consider Bank Conflicts**: Pad shared memory arrays when necessary to avoid conflicts
2. ** Tile Size Matters**: Balance shared memory usage with occupancy requirements
3. ** Double Buffering**: Overlap computation with memory loading for maximum performance
4. ** Profile Regularly**: Use Nsight Compute to validate bank conflict elimination
5. ** Pattern-Specific Optimization**: Tailor shared memory layouts to your access patterns

##  **Related Guides**

- **Next Step**: [Constant Memory Complete Guide](3d_constant_memory.md) - Optimize read-only data access
- **Previous**: [Global Memory Advanced Guide](3b_global_memory.md) - Coalescing fundamentals
- **Debugging**: [Memory Debugging Toolkit](3f_memory_debugging.md) - Troubleshoot shared memory issues
- **Overview**: [Memory Hierarchy Overview](3_cuda_memory_hierarchy.md) - Quick reference and navigation

---

** Pro Tip**: Start with basic tiling, eliminate bank conflicts, then consider advanced techniques like double buffering. Measure performance at each step to validate improvements!
