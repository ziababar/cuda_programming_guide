#pragma once

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

/**
 * Asynchronous Memory Copy (cp.async) Demonstration
 *
 * This file demonstrates using `cuda::pipeline` (introduced in CUDA 11) to manage
 * asynchronous data copies from Global Memory to Shared Memory.
 *
 * Key Concepts:
 * - cp.async: Bypasses register file, moving data directly from L2 to Shared Memory.
 * - cuda::pipeline: C++ interface for managing async stages.
 * - Double Buffering: Overlapping the computation of the current batch with the loading of the next.
 *
 * Requirements:
 * - Compute Capability 8.0+ (Ampere or newer)
 * - CUDA 11.0+
 */

// Double buffering kernel using cuda::pipeline
template <int BLOCK_SIZE>
__global__ void async_copy_kernel(float* d_out, const float* d_in, int N) {
    // 1. Allocate Shared Memory (Double Buffer)
    // We need 2 buffers of size BLOCK_SIZE
    __shared__ float buffer[2][BLOCK_SIZE];

    auto block = cg::this_thread_block();

    // 2. Initialize Pipeline
    // cuda::thread_scope_block means the pipeline barrier synchronizes the whole block
    cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;

    // Number of full batches to process per block
    // (Simplified for demo: assuming N is divisible by grid size)
    // In a grid-stride loop, logic would be slightly more complex.
    // Here we assume 1 block processes 1 chunk of N elements just for the snippet flow.
    // Let's implement a Grid-Stride Loop for robustness.

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Current buffer index (0 or 1)
    int view = 0;

    // Prologue: Issue the first batch load
    // -----------------------------------------------------------
    pipe.producer_acquire();
    if (global_tid < N) {
        // Asynchronously copy from Global(d_in) to Shared(buffer[view])
        cuda::memcpy_async(&buffer[view][tid], &d_in[global_tid], sizeof(float), pipe);
    }
    // Commit the "batch" of copies. This marks a stage in the pipeline.
    pipe.producer_commit();

    // Main Loop
    // -----------------------------------------------------------
    for (int i = global_tid; i < N; i += stride) {
        // We are processing the batch loaded at 'i' (currently in buffer[view])
        // We want to pre-fetch the NEXT batch (at i + stride) into buffer[view ^ 1]

        int next_i = i + stride;

        // Issue Next Copy (Prologue for next iteration)
        pipe.producer_acquire();
        if (next_i < N) {
             cuda::memcpy_async(&buffer[view ^ 1][tid], &d_in[next_i], sizeof(float), pipe);
        }
        pipe.producer_commit();

        // Wait for Current Copy to Complete
        // wait_prior<1>: Wait until only the most recent 1 stage is incomplete.
        // Since we just issued 'next' (stage N+1), this waits for 'current' (stage N) to finish.
        pipe.consumer_wait();

        // Block Sync: Ensure all threads in block have their data ready in shared mem
        block.sync();

        // Compute Phase: Read from Shared Memory, Write to Global
        if (i < N) {
            float val = buffer[view][tid];
            // Simple math operation
            d_out[i] = val * 2.0f + 1.0f;
        }

        // Release the stage so the buffer can be reused
        pipe.consumer_release();

        // Swap buffers
        view ^= 1;
    }
}

// Host wrapper to invoke the kernel
inline void run_async_copy_demo(float* d_out, float* d_in, int N) {
    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("Launching Async Copy Kernel with %d blocks of %d threads...\n", numBlocks, blockSize);

    // Check for Ampere or newer
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        printf("Skipping Async Copy: Requires Compute Capability 8.0+ (Detected %d.%d)\n", prop.major, prop.minor);
        return;
    }

    async_copy_kernel<128><<<numBlocks, blockSize>>>(d_out, d_in, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launched successfully.\n");
    }
}
