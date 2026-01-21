#pragma once

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

/**
 * @brief Demonstrates Asynchronous Data Copy (Global -> Shared) using cuda::pipeline.
 *
 * This feature (Compute Capability 8.0+) allows threads to issue memory loads
 * that bypass the register file, going directly from Global to Shared Memory.
 * This frees up registers and allows for overlap of compute and data movement.
 */
template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void async_copy_pipeline_kernel(float* d_in, float* d_out, int N) {
    // Shared memory for double buffering (2 stages)
    // We need 2 buffers: one for the data being consumed, one for the data being loaded
    __shared__ float smem_buffer[2][BLOCK_SIZE * TILE_SIZE];

    auto block = cg::this_thread_block();
    auto tid = block.thread_rank();

    // Create a pipeline object with 2 stages
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Calculate number of batches
    int num_batches = (N + (BLOCK_SIZE * TILE_SIZE) - 1) / (BLOCK_SIZE * TILE_SIZE);

    // Prologue: Pre-load the first batch
    int batch_idx = 0;
    int global_idx = batch_idx * (BLOCK_SIZE * TILE_SIZE) + tid * TILE_SIZE;

    // Acquire stage 0 resources
    pipe.producer_acquire();

    // Issue async copies for the first batch
    if (global_idx < N) {
        // Copy TILE_SIZE elements per thread
        for (int i = 0; i < TILE_SIZE; ++i) {
            if (global_idx + i < N) {
                // cp.async: Global -> Shared
                // Note: cuda::memcpy_async matches the pipeline interface
                cuda::memcpy_async(&smem_buffer[0][tid * TILE_SIZE + i],
                                   &d_in[global_idx + i],
                                   sizeof(float),
                                   pipe);
            }
        }
    }

    // Commit the issued commands for stage 0
    pipe.producer_commit();

    // Main Loop
    for (; batch_idx < num_batches; ++batch_idx) {
        // Index for the next batch (to be loaded)
        int next_batch_idx = batch_idx + 1;
        int buffer_idx = batch_idx % 2;
        int next_buffer_idx = next_batch_idx % 2;

        // 1. Issue Next Batch (if exists)
        if (next_batch_idx < num_batches) {
            int next_global_idx = next_batch_idx * (BLOCK_SIZE * TILE_SIZE) + tid * TILE_SIZE;

            pipe.producer_acquire();
            if (next_global_idx < N) {
                for (int i = 0; i < TILE_SIZE; ++i) {
                    if (next_global_idx + i < N) {
                        cuda::memcpy_async(&smem_buffer[next_buffer_idx][tid * TILE_SIZE + i],
                                           &d_in[next_global_idx + i],
                                           sizeof(float),
                                           pipe);
                    }
                }
            }
            pipe.producer_commit();
        }

        // 2. Wait for Current Batch (Compute)
        // We wait for the oldest stage to complete.
        pipe.consumer_wait();

        // Perform computation on the data in shared memory (smem_buffer[buffer_idx])
        // Sync block to ensure all threads have their data ready in shared memory
        // Note: pipeline wait only ensures *this* thread's copy is done.
        // For shared memory visibility across the block, we typically need block.sync(),
        // but here each thread accesses its OWN part of shared memory (private tile),
        // so strictly speaking, block sync isn't needed if we only process our own data.
        // However, usually we process data loaded by others (stencil/reduction).
        // For this simple example (vector copy/scale), we stick to thread-local tile.

        block.sync(); // Good practice if access pattern changes

        // Compute: Scale the data
        int curr_global_idx = batch_idx * (BLOCK_SIZE * TILE_SIZE) + tid * TILE_SIZE;
        for (int i = 0; i < TILE_SIZE; ++i) {
            if (curr_global_idx + i < N) {
                float val = smem_buffer[buffer_idx][tid * TILE_SIZE + i];
                // Expensive math operation simulation
                val = val * 2.0f;
                d_out[curr_global_idx + i] = val;
            }
        }

        block.sync(); // Wait for everyone to finish reading before we overwrite in next loop

        // Release the stage
        pipe.consumer_release();
    }
}

inline void run_async_copy_demo(float* d_in, float* d_out, int N) {
    // Must check for Compute Capability >= 8.0
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 8) {
        printf("Skipping Async Copy Demo: Requires Compute Capability 8.0+ (Detected %d.%d)\n", prop.major, prop.minor);
        return;
    }

    const int BLOCK_SIZE = 128;
    const int TILE_SIZE = 4; // Each thread handles 4 floats

    // Calculate grid size
    int threads_per_batch = BLOCK_SIZE * TILE_SIZE;
    // We launch enough blocks, but the kernel loop handles chunks too.
    // For simplicity, launch 1 block to process everything or grid strided.
    // The kernel is written as a grid-strided loop over "batches" but hardcoded for 1 block logic in the example for simplicity.
    // Let's launch 1 block to demonstrate the pipeline loop clearly.

    async_copy_pipeline_kernel<BLOCK_SIZE, TILE_SIZE><<<1, BLOCK_SIZE>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Async Copy Kernel Launched.\n");
    }
}
