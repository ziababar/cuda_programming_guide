#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cmath>

namespace cg = cooperative_groups;

/**
 * @brief Demonstrates Asynchronous Memory Copy (cp.async) using cuda::pipeline
 *
 * Requires Compute Capability 8.0+ (Ampere)
 */
class AsyncCopyDemo {
public:
    // Kernel using cuda::pipeline for async copy
    // Concept: Double buffering with cp.async
    // 1. Issue copy for batch i
    // 2. Wait for batch i-1
    // 3. Compute batch i-1
    // 4. Repeat
    __global__ static void async_copy_pipeline_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
        extern __shared__ float shared_mem[];
        float* smem_buffer_0 = shared_mem; // Buffer A
        float* smem_buffer_1 = shared_mem + blockDim.x; // Buffer B

        // Pipeline object
        cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

        int tid = threadIdx.x;

        // Grid-Stride Loop:
        // Each block processes a chunk of data. To demonstrate pipelining,
        // we process 'elements_per_tile' (blockDim.x) elements per iteration.
        // The total number of tiles is N / blockDim.x.
        // We stride by gridDim.x to ensure coverages.

        int elements_per_tile = blockDim.x;
        int num_total_tiles = (N + elements_per_tile - 1) / elements_per_tile;

        for (int base_tile_idx = blockIdx.x; base_tile_idx < num_total_tiles; base_tile_idx += gridDim.x) {

            // For this specific 'chunk' assigned to the block, we might want to loop internally
            // if we were processing a large contiguous chunk.
            // But standard grid-stride is usually one tile per iteration of the main loop.
            // To demonstrate pipeline *latency hiding*, we need a loop inside.
            // However, the grid-stride loop *itself* is the loop we pipeline!

            // Let's pipeline the grid-stride loop.
            // We need to manage the indices carefully.

            // Prologue: Issue copy for the FIRST tile of this thread's workload
            // But wait, grid-stride is irregular.
            // Easier approach for demo: Block-Linear Loop.
            // Each block processes a contiguous chunk of size (TilesPerBlock * BlockDim).
            // But let's stick to Grid Stride for robustness, but we need to know the 'next' tile.

            // Pipeline state:
            // We need to issue 'next' before waiting for 'current'.

            // Let's effectively unroll the grid stride loop slightly for the pipeline.

            // Current index we are processing
            int current_tile_idx = base_tile_idx;

            // Issue Load for First Tile
            if (current_tile_idx < num_total_tiles) {
                 pipe.producer_acquire();
                 int src_idx = current_tile_idx * elements_per_tile + tid;
                 if (src_idx < N) {
                     cuda::memcpy_async(&smem_buffer_0[tid], &input[src_idx], sizeof(float), pipe);
                 }
                 pipe.producer_commit();
            }

            // We need to process the loop.
            // Since 'base_tile_idx' increments by gridDim.x, the 'next' tile is base_tile_idx + gridDim.x.
            // But we are inside the `for` loop which does that increment.
            // This makes pipelining inside a standard `for` loop tricky without restructuring.

            // Let's restructure:
            // Loop while we have work.
            // Using a specific structure for 2-stage pipeline.

            // NOTE: We cannot easily pipeline a grid-stride loop if the stride is large and we want to keep data in registers/smem.
            // But here we are just buffering tiles.

            // Let's restart the loop logic.
            // We will iterate `i` from `blockIdx.x` to `num_total_tiles` with stride `gridDim.x`.
            // Stage 0: Load `i`
            // Stage 1: Load `i + gridDim.x`, Compute `i`
            // Stage 2: Load `i + 2*gridDim.x`, Compute `i + gridDim.x`
            // ...

            // Wait, standard pipeline is:
            // Prologue: Load Tile 0
            // Loop k=0 to M:
            //    Load Tile k+1
            //    Wait Tile k
            //    Compute Tile k
            // Epilogue: Wait/Compute last

        }

        // Re-implementing with explicit loop structure for clarity and correctness
        // We will process tiles: blockIdx.x, blockIdx.x + grid, blockIdx.x + 2*grid...

        int tile_idx = blockIdx.x;

        // Prologue
        if (tile_idx < num_total_tiles) {
             pipe.producer_acquire();
             int src_idx = tile_idx * elements_per_tile + tid;
             if (src_idx < N) {
                 cuda::memcpy_async(&smem_buffer_0[tid], &input[src_idx], sizeof(float), pipe);
             } else {
                 // Pad with 0 if out of bounds (shouldn't happen with logic above but good safety)
             }
             pipe.producer_commit();
        }

        // Main Loop
        // We need to track which buffer holds the current data.
        // buffer_ptr points to the buffer we just loaded into (and will compute on next)
        // But for double buffering, we load into Next, compute on Current.

        // Let's use an index `i` to count iterations for buffer swapping.
        int iter = 0;

        while (tile_idx < num_total_tiles) {
             // Next tile index
             int next_tile_idx = tile_idx + gridDim.x;

             // Issue Copy for Next Tile (if exists)
             if (next_tile_idx < num_total_tiles) {
                 pipe.producer_acquire();
                 int src_idx = next_tile_idx * elements_per_tile + tid;
                 // Determine buffer for next load: (iter + 1) % 2
                 float* dest_ptr = ((iter + 1) % 2 == 0) ? smem_buffer_0 : smem_buffer_1;

                 if (src_idx < N) {
                     cuda::memcpy_async(&dest_ptr[tid], &input[src_idx], sizeof(float), pipe);
                 }
                 pipe.producer_commit();
             }

             // Wait for Current Tile
             pipe.consumer_wait();

             // Compute Current Tile
             float* src_ptr = (iter % 2 == 0) ? smem_buffer_0 : smem_buffer_1;
             float val = src_ptr[tid]; // Should be valid as we masked load

             // Logic: if we are out of bounds (N), we shouldn't compute/store.
             // But tile logic handles 'tile_idx'. We need to check element bounds for store.
             int global_idx = tile_idx * elements_per_tile + tid;

             if (global_idx < N) {
                 val = val * val; // Square
                 output[global_idx] = val;
             }

             // Release Current Tile stage
             pipe.consumer_release();

             // Advance
             tile_idx = next_tile_idx;
             iter++;
        }
    }

    // Host function to run the demo
    static void run_demo() {
        int N = 1024 * 1024;
        size_t bytes = N * sizeof(float);

        float *h_input, *h_output;
        float *d_input, *d_output;

        // Use cudaHostAlloc for pinned memory
        cudaHostAlloc(&h_input, bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_output, bytes, cudaHostAllocDefault);

        for(int i=0; i<N; ++i) h_input[i] = 1.0f * i;

        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);

        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        // Fix the number of blocks to ensure looping.
        // N = 1M. threads = 256. Total tiles = 4096.
        // If we use 80 blocks (typical SM count on A100 is 108), each block does ~50 iterations.
        int blocks = 80;

        size_t smem_size = 2 * threads * sizeof(float);

        printf("Running Async Copy Pipeline Kernel...\n");
        printf("Grid: %d blocks, %d threads. Total Elements: %d\n", blocks, threads, N);

        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);

        if (props.major < 8) {
            printf("Skipping Async Copy Demo: Requires Compute Capability 8.0+ (Detected %d.%d)\n", props.major, props.minor);
        } else {
             async_copy_pipeline_kernel<<<blocks, threads, smem_size>>>(d_input, d_output, N);
             cudaDeviceSynchronize();

             cudaError_t err = cudaGetLastError();
             if (err != cudaSuccess) {
                 printf("CUDA Error: %s\n", cudaGetErrorString(err));
             } else {
                 printf("Kernel executed successfully.\n");
             }
        }

        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        // Verify
        bool correct = true;
        for(int i=0; i<N; ++i) {
            float expected = (1.0f * i) * (1.0f * i);
            if (std::abs(h_output[i] - expected) > 1e-5) {
                correct = false;
                printf("Mismatch at %d: Expected %f, Got %f\n", i, expected, h_output[i]);
                break; // Stop at first error
            }
        }

        if (correct) printf("Verification PASSED\n");
        else printf("Verification FAILED\n");

        cudaFreeHost(h_input);
        cudaFreeHost(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
