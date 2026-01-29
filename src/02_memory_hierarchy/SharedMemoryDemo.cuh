#pragma once

#include <cuda_runtime.h>
#include <cstdio>

namespace SharedMemoryDemo {

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

template<int TILE_SIZE>
__device__ constexpr int calculate_padding() {
    // Ensure no bank conflicts for square tiles
    return (32 % TILE_SIZE == 0) ? 1 : 0;
}

// -------------------------------------------------------------------------
// Matrix Multiplication Kernels
// -------------------------------------------------------------------------

// Level 1: Basic Tiling Implementation
#define TILE_SIZE_BASIC 16
__global__ void matmul_tiled_basic(float* A, float* B, float* C, int N) {
    // Shared memory tiles
    __shared__ float tile_A[TILE_SIZE_BASIC][TILE_SIZE_BASIC];
    __shared__ float tile_B[TILE_SIZE_BASIC][TILE_SIZE_BASIC];

    int row = blockIdx.y * TILE_SIZE_BASIC + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_BASIC + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE_BASIC - 1) / TILE_SIZE_BASIC; ++tile) {
        // Load tile from A
        if (row < N && tile * TILE_SIZE_BASIC + threadIdx.x < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE_BASIC + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B
        if (col < N && tile * TILE_SIZE_BASIC + threadIdx.y < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE_BASIC + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_SIZE_BASIC; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Store result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Level 2: Bank Conflict-Free Version
#define TILE_SIZE_NC 16
#define PADDING_NC 1  // Avoid bank conflicts

__global__ void matmul_no_conflicts(float* A, float* B, float* C, int N) {
    // Add padding to avoid bank conflicts
    __shared__ float tile_A[TILE_SIZE_NC][TILE_SIZE_NC + PADDING_NC];
    __shared__ float tile_B[TILE_SIZE_NC][TILE_SIZE_NC + PADDING_NC];

    int row = blockIdx.y * TILE_SIZE_NC + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_NC + threadIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE_NC - 1) / TILE_SIZE_NC; ++tile) {
        // Coalesced loading with bounds checking
        int a_col = tile * TILE_SIZE_NC + tx;
        int b_row = tile * TILE_SIZE_NC + ty;

        tile_A[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        tile_B[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Unrolled computation for better performance
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_NC; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Level 3: Double Buffering for Maximum Performance
#define TILE_SIZE_DB 16
#define PADDING_DB 1

__global__ void matmul_double_buffered(float* A, float* B, float* C, int N) {
    // Double buffering: 2 sets of tiles
    __shared__ float tile_A[2][TILE_SIZE_DB][TILE_SIZE_DB + PADDING_DB];
    __shared__ float tile_B[2][TILE_SIZE_DB][TILE_SIZE_DB + PADDING_DB];

    int row = blockIdx.y * TILE_SIZE_DB + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_DB + threadIdx.x;
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE_DB - 1) / TILE_SIZE_DB;

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
            int next_tile_offset = (tile + 1) * TILE_SIZE_DB;
            int a_col = next_tile_offset + tx;
            int b_row = next_tile_offset + ty;

            tile_A[next_buffer][ty][tx] = (row < N && a_col < N) ?
                A[row * N + a_col] : 0.0f;
            tile_B[next_buffer][ty][tx] = (b_row < N && col < N) ?
                B[b_row * N + col] : 0.0f;
        }

        // Compute using current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_DB; ++k) {
            sum += tile_A[current_buffer][ty][k] * tile_B[current_buffer][k][tx];
        }

        current_buffer = next_buffer;
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// -------------------------------------------------------------------------
// Optimization & Tiling Patterns
// -------------------------------------------------------------------------

// Array Reordering (Transpose)
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

// Rectangular Tiling
#define TILE_WIDTH_RECT 32
#define TILE_HEIGHT_RECT 8

__global__ void rectangular_tiling_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_HEIGHT_RECT][TILE_WIDTH_RECT + 1];
    __shared__ float tile_B[TILE_WIDTH_RECT][TILE_HEIGHT_RECT + 1];

    int row = blockIdx.y * TILE_HEIGHT_RECT + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH_RECT + threadIdx.x;

    float sum = 0.0f;

    // Optimized for specific matrix dimensions and memory hierarchy
    for (int tile = 0; tile < (K + TILE_WIDTH_RECT - 1) / TILE_WIDTH_RECT; ++tile) {
        // Load rectangular tiles optimized for memory coalescing
        if (row < M && tile * TILE_WIDTH_RECT + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_WIDTH_RECT + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tile * TILE_WIDTH_RECT + threadIdx.y < K && col < N) {
            tile_B[threadIdx.x][threadIdx.y] = B[(tile * TILE_WIDTH_RECT + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH_RECT; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Multi-Level Tiling
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

// Register Optimization
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

// Memory Access Pattern Optimization
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

// -------------------------------------------------------------------------
// Real-World Applications
// -------------------------------------------------------------------------

// Scientific Computing: Stencil Operations
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

// Graphics: Fast Gaussian Blur
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
            // Manual min/max to avoid <algorithm> dependency
            src_x = (src_x < 0) ? 0 : ((src_x > width - 1) ? width - 1 : src_x);
            src_y = (src_y < 0) ? 0 : ((src_y > height - 1) ? height - 1 : src_y);

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

} // namespace SharedMemoryDemo
