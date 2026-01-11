#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <vector>

// Helper for checking CUDA errors
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        } \
    } while (0)
#endif

// Kernel demonstrating cp.async via cuda::pipeline
// Each block processes a distinct tile of the input array.
// Block Size: Fixed at compile time for shared memory allocation
template <int BLOCK_SIZE>
__global__ void pipeline_kernel(float* global_in, float* global_out, int N) {
    // Pipeline object
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Shared memory buffers for double buffering
    // 2 stages: [0] for loading, [1] for computing (or vice versa)
    __shared__ float shared_buffer[2][BLOCK_SIZE];

    int tid = threadIdx.x;

    // Calculate the global offset for this block
    // Each block processes BLOCK_SIZE elements * iterations_per_block (if we loop)
    // Here, for simplicity, we map 1 block to 1 tile of data.
    int block_offset = blockIdx.x * BLOCK_SIZE;

    // Check if this block is within bounds
    if (block_offset >= N) return;

    // Load the FIRST batch (Prologue)
    // We only have 1 batch per block in this simple 1:1 mapping.

    int global_idx = block_offset + tid;

    // Stage 0: Issue copy for the current tile
    pipe.producer_acquire();
    if (global_idx < N) {
        cuda::memcpy_async(&shared_buffer[0][tid], &global_in[global_idx], sizeof(float), pipe);
    }
    pipe.producer_commit();

    // Stage 1: Wait and Compute
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    pipe.consumer_release();

    if (global_idx < N) {
        float val = shared_buffer[0][tid];
        val *= 2.0f;
        global_out[global_idx] = val;
    }
}

// Optimized Kernel: Processing multiple tiles per block (Grid-Stride Loop with Pipeline)
// This implements the optimal pipeline pattern:
// 1. Wait for batch K (making it ready for compute)
// 2. Issue batch K+1 (starting the asynchronous fetch)
// 3. Compute batch K (overlapping with fetch of K+1)
template <int BLOCK_SIZE>
__global__ void pipeline_kernel_optimized(float* global_in, float* global_out, int N) {
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    __shared__ float shared_buffer[2][BLOCK_SIZE];

    int tid = threadIdx.x;
    int grid_stride = gridDim.x * BLOCK_SIZE;

    // PROLOGUE: Start loading Tile 0
    int idx_k = blockIdx.x * BLOCK_SIZE + tid;

    pipe.producer_acquire();
    if (idx_k < N) {
        cuda::memcpy_async(&shared_buffer[0][tid], &global_in[idx_k], sizeof(float), pipe);
    }
    pipe.producer_commit();

    // Main Loop
    for (int k = 0; ; ++k) {
        // 1. Wait for batch K
        // We need batch K to be ready before we compute on it.
        // wait_prior<0> waits until all older batches are done. Since we only have 1 active (K), this waits for K.
        cuda::pipeline_consumer_wait_prior<0>(pipe);
        pipe.consumer_release();

        // 2. Issue Load for K+1 (if exists)
        // By issuing this AFTER waiting for K, we ensure we don't block the compute of K.
        // This allows the memory load of K+1 to overlap with the computation of K.

        int idx_next = idx_k + grid_stride;
        bool has_next = (idx_next < N) || (idx_next - tid < N); // Check bounds loosely

        if (has_next) {
             pipe.producer_acquire();
             if (idx_next < N) {
                 cuda::memcpy_async(&shared_buffer[(k+1)%2][tid], &global_in[idx_next], sizeof(float), pipe);
             }
             pipe.producer_commit();
        }

        // 3. Compute K
        // The data for K is in shared_buffer[k%2].
        if (idx_k < N) {
            float val = shared_buffer[k%2][tid];
            val *= 2.0f;
            global_out[idx_k] = val;
        }

        // Break if no next tile was loaded
        if (!has_next) break;

        // Advance
        idx_k = idx_next;
    }
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Copy (cp.async) Demo ===\n");

        int N = 1024 * 1024;
        size_t bytes = N * sizeof(float);

        // Host memory
        std::vector<float> h_in(N, 1.0f);
        std::vector<float> h_out(N, 0.0f);

        // Device memory
        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, bytes));
        CHECK_CUDA(cudaMalloc(&d_out, bytes));

        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

        // Launch Kernel
        // Use enough blocks to cover the array
        int block_size = 128;
        int grid_size = (N + block_size - 1) / block_size;

        // To demonstrate the pipeline loop, we can use a smaller grid size
        // so each block processes multiple tiles.
        // Let's use 1/4th the necessary blocks, forcing 4 iterations per block.
        int sm_count = 80; // Assume 80 SMs
        int optimized_grid_size = sm_count * 4; // High enough occupancy
        if (optimized_grid_size > grid_size) optimized_grid_size = grid_size;

        printf("Launching Optimized Pipeline Kernel with Grid=%d, Block=%d (Grid Stride Loop)\n", optimized_grid_size, block_size);
        pipeline_kernel_optimized<128><<<optimized_grid_size, block_size>>>(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Verify
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < N; ++i) {
            if (abs(h_out[i] - 2.0f) > 1e-5) {
                printf("Mismatch at %d: Expected 2.0, got %f\n", i, h_out[i]);
                correct = false;
                break;
            }
        }

        if (correct) {
            printf("Success! All values computed correctly.\n");
        } else {
            printf("Verification failed.\n");
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
