#pragma once

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>
#include <cmath>

namespace cg = cooperative_groups;

// --------------------------------------------------------------------------------
// Asynchronous Memory Copy (cp.async) Demo
// --------------------------------------------------------------------------------
// This example demonstrates how to use the cuda::pipeline API (Compute Capability 8.0+)
// to overlap global memory loads with computation using Shared Memory double buffering.
//
// Concepts:
// 1. cuda::pipeline: Manages the state of asynchronous copies.
// 2. cuda::memcpy_async: Initiates a copy from global to shared memory without blocking.
// 3. Double Buffering: Using two buffers to hide latency (Compute on A while loading B).
// --------------------------------------------------------------------------------

// Improved Kernel: Process a sequence of tiles to demonstrate pipelining
template <int BlockSize, int TileSize>
__global__ void async_pipeline_demo_kernel(const float* __restrict__ input, float* __restrict__ output, int num_tiles) {
    extern __shared__ float smem[];
    float* buffers[2];
    buffers[0] = smem;
    buffers[1] = smem + TileSize;

    auto block = cg::this_thread_block();
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Index of the tile this block starts with
    int start_tile = blockIdx.x * num_tiles; // Each block processes 'num_tiles' tiles
    int end_tile = start_tile + num_tiles;

    // Current buffer index (0 or 1)
    int view_idx = 0;

    // Prologue: Start loading the first tile
    pipe.producer_acquire();
    size_t global_offset = (size_t)start_tile * TileSize;
    for (int i = threadIdx.x; i < TileSize; i += BlockSize) {
        cuda::memcpy_async(&buffers[view_idx][i], &input[global_offset + i], sizeof(float), pipe);
    }
    pipe.producer_commit();

    // Loop over remaining tiles
    for (int t = start_tile; t < end_tile - 1; ++t) {
        // Prepare next buffer index
        int next_view_idx = 1 - view_idx;

        // Issue load for next tile (t + 1)
        pipe.producer_acquire();
        size_t next_global_offset = (size_t)(t + 1) * TileSize;
        for (int i = threadIdx.x; i < TileSize; i += BlockSize) {
            cuda::memcpy_async(&buffers[next_view_idx][i], &input[next_global_offset + i], sizeof(float), pipe);
        }
        pipe.producer_commit();

        // Wait for the current tile (t) to finish loading
        pipe.consumer_wait();

        // Synchronize threads to ensure everyone's data is ready (if needed for cross-thread access)
        // For element-wise, technically not needed if mapping is 1:1, but good practice.
        block.sync();

        // Compute on current tile (t) residing in buffers[view_idx]
        size_t out_offset = (size_t)t * TileSize;
        for (int i = threadIdx.x; i < TileSize; i += BlockSize) {
            // Simple compute: square the value
            float val = buffers[view_idx][i];
            output[out_offset + i] = val * val;
        }

        // Done with current buffer
        block.sync();
        pipe.consumer_release();

        // Switch view
        view_idx = next_view_idx;
    }

    // Epilogue: Process the final tile
    pipe.consumer_wait();
    block.sync();

    size_t out_offset = (size_t)(end_tile - 1) * TileSize;
    for (int i = threadIdx.x; i < TileSize; i += BlockSize) {
        float val = buffers[view_idx][i];
        output[out_offset + i] = val * val;
    }
    pipe.consumer_release();
}

// Host wrapper
inline void run_async_copy_demo(int num_blocks, int tiles_per_block) {
    const int BlockSize = 128;
    const int TileSize = 512; // 4 float per thread if 128 threads

    int total_tiles = num_blocks * tiles_per_block;
    size_t N = (size_t)total_tiles * TileSize;
    size_t bytes = N * sizeof(float);

    // Host allocations (pinned for async safety, though not strictly required for device-side async copy)
    float *h_in, *h_out;
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);

    for (size_t i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Shared memory size: 2 buffers * TileSize * sizeof(float)
    size_t smem_size = 2 * TileSize * sizeof(float);

    printf("Launching Async Copy Pipeline Kernel...\n");
    printf("Grid: %d, Block: %d, Tiles/Block: %d, SharedMem: %zu bytes\n",
           num_blocks, BlockSize, tiles_per_block, smem_size);

    async_pipeline_demo_kernel<BlockSize, TileSize>
        <<<num_blocks, BlockSize, smem_size>>>(d_in, d_out, tiles_per_block);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        cudaDeviceSynchronize();
        printf("Kernel completed successfully.\n");
    }

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(h_out[i] - 1.0f) > 1e-5) {
            correct = false;
            printf("Mismatch at %zu: %f\n", i, h_out[i]);
            break;
        }
    }
    printf("Verification: %s\n", correct ? "PASS" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
}
