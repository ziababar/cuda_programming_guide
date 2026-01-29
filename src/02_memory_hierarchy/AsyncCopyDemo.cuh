#pragma once

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>

namespace AsyncCopyDemo {

namespace cg = cooperative_groups;

// -------------------------------------------------------------------------
// Asynchronous Copy Pipeline Demo
// -------------------------------------------------------------------------
// This kernel demonstrates how to use cuda::pipeline to overlap
// global-to-shared memory copies with computation.
//
// Requirements: Compute Capability 8.0+ (Ampere)

template <int BLOCK_SIZE>
__global__ void async_copy_pipeline_kernel(float* global_in, float* global_out, int N, int batches_per_block) {
    // Shared memory buffer
    // We use double buffering: 2 stages
    __shared__ float shared_buffer[2][BLOCK_SIZE];

    auto block = cg::this_thread_block();
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    int tid = threadIdx.x;

    // Each block processes 'batches_per_block' chunks of size BLOCK_SIZE
    int block_start_idx = blockIdx.x * batches_per_block * BLOCK_SIZE;

    // Prologue: Load the first batch (Batch 0)
    int first_batch_global_idx = block_start_idx + tid;

    pipeline.producer_acquire();
    if (first_batch_global_idx < N) {
        // Asynchronous copy from global to shared memory
        cuda::memcpy_async(&shared_buffer[0][tid], &global_in[first_batch_global_idx], sizeof(float), pipeline);
    }
    pipeline.producer_commit();

    // Loop over the batches assigned to this block
    for (int i = 0; i < batches_per_block; ++i) {
        int buffer_idx = i % 2;
        int next_buffer_idx = (i + 1) % 2;

        // Prefetch next batch (Batch i+1) if it's not the last one
        if (i < batches_per_block - 1) {
            int next_batch_global_idx = block_start_idx + (i + 1) * BLOCK_SIZE + tid;

            pipeline.producer_acquire();
            if (next_batch_global_idx < N) {
                cuda::memcpy_async(&shared_buffer[next_buffer_idx][tid], &global_in[next_batch_global_idx], sizeof(float), pipeline);
            }
            pipeline.producer_commit();
        }

        // Wait for current batch (Batch i) to be ready
        pipeline.consumer_wait();

        // Compute on current batch
        float val = shared_buffer[buffer_idx][tid];
        val = val * 2.0f; // Simple computation

        // Store result
        int current_global_idx = block_start_idx + i * BLOCK_SIZE + tid;
        if (current_global_idx < N) {
            global_out[current_global_idx] = val;
        }

        // Release the buffer
        pipeline.consumer_release();
    }
}

// -------------------------------------------------------------------------
// Host Helper
// -------------------------------------------------------------------------
// This function helps to launch the kernel and manage memory.

inline void run_async_copy_demo(int N) {
    int size = N * sizeof(float);
    float *h_in, *h_out;
    float *d_in, *d_out;

    // Use pinned memory for host buffers (best practice)
    cudaHostAlloc(&h_in, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_out, size, cudaHostAllocDefault);

    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch configuration
    constexpr int BLOCK_SIZE = 128;
    // We want each block to process, say, 4 batches to demonstrate pipelining
    int batches_per_block = 4;
    int items_per_block = BLOCK_SIZE * batches_per_block;

    // Calculate grid size needed to cover N
    int grid_size = (N + items_per_block - 1) / items_per_block;

    // Check device properties for Ampere support
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major >= 8) {
        printf("Launching Async Copy Kernel on SM %d.%d...\n", prop.major, prop.minor);
        async_copy_pipeline_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N, batches_per_block);
        cudaDeviceSynchronize();
    } else {
        printf("Skipping Async Copy Demo: Requires Compute Capability 8.0+ (Found %d.%d)\n", prop.major, prop.minor);
    }

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Verify
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != h_in[i] * 2.0f) {
            correct = false;
            printf("Error at index %d: expected %f, got %f\n", i, h_in[i] * 2.0f, h_out[i]);
            break;
        }
    }

    if (correct) {
        printf("Async Copy Demo: SUCCESS\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
}

} // namespace AsyncCopyDemo
