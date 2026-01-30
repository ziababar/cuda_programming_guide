#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cstdio>

namespace cg = cooperative_groups;

/**
 * Async Copy Demo (Compute Capability 8.0+)
 *
 * Demonstrates using cuda::pipeline to overlap Global->Shared memory copies
 * with computation. This bypasses the register file for loads, reducing
 * register pressure and hiding memory latency.
 */

// Pipeline stages
#define PIPELINE_STAGES 2

template <int BLOCK_SIZE>
__global__ void async_copy_kernel(float* d_out, const float* d_in, int N, int repeat_compute) {
    // Shared memory for 2 stages (double buffering)
    // Tiled partition size per stage = BLOCK_SIZE
    extern __shared__ float shared_mem[];
    float* smem_buffers[PIPELINE_STAGES];
    smem_buffers[0] = shared_mem;
    smem_buffers[1] = shared_mem + BLOCK_SIZE;

    auto block = cg::this_thread_block();
    int tid = block.thread_rank();

    // Create pipeline object
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Total batches to process
    // Each block processes a distinct chunk of the array (simplification for demo)
    // In a real kernel, this would be a grid-stride loop
    int batches_per_block = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block_offset = 0; // Assuming 1 block for simplicity or standard indexing

    // Prologue: Load first batch
    // Acquire a stage in the pipeline
    pipe.producer_acquire();

    // Async copy from Global to Shared
    // Note: cuda::memcpy_async matches the standard memcpy signature
    if (tid < BLOCK_SIZE && tid < N) {
        cuda::memcpy_async(smem_buffers[0] + tid, d_in + tid, sizeof(float), pipe);
    }

    // Commit the stage (signal that we are done issuing copies for this stage)
    pipe.producer_commit();

    // Main Loop
    for (int i = 0; i < batches_per_block; ++i) {
        int next_batch = i + 1;
        int curr_stage = i % PIPELINE_STAGES;
        int next_stage = next_batch % PIPELINE_STAGES;
        int next_offset = next_batch * BLOCK_SIZE;

        // 1. Issue Copy for NEXT batch (if valid)
        if (next_batch < batches_per_block) {
            pipe.producer_acquire();
            if (tid < BLOCK_SIZE && (next_offset + tid) < N) {
                cuda::memcpy_async(smem_buffers[next_stage] + tid,
                                 d_in + next_offset + tid,
                                 sizeof(float), pipe);
            }
            pipe.producer_commit();
        }

        // 2. Wait for CURRENT batch to finish loading
        pipe.consumer_wait();

        // 3. Compute on CURRENT batch (now in shared memory)
        float val = 0.0f;
        if (tid < BLOCK_SIZE && (i * BLOCK_SIZE + tid) < N) {
            val = smem_buffers[curr_stage][tid];

            // Heavy computation to hide latency of next load
            for (int k = 0; k < repeat_compute; ++k) {
                val = val * 1.01f + 0.01f;
            }
        }

        // 4. Release CURRENT stage buffer for reuse
        pipe.consumer_release();

        // Write result
        if (tid < BLOCK_SIZE && (i * BLOCK_SIZE + tid) < N) {
            d_out[i * BLOCK_SIZE + tid] = val;
        }
    }
}

// Host wrapper class
class AsyncCopyDemo {
public:
    static void run_demo(int N, int compute_load) {
        printf("Running Async Copy Demo (N=%d)...\n", N);

        // Check Compute Capability
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (prop.major < 8) {
            printf("Skipping: Async Copy requires Compute Capability 8.0+ (Ampere)\n");
            printf("Current device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
            return;
        }

        float *h_in, *h_out, *d_in, *d_out;
        size_t bytes = N * sizeof(float);

        // Host pinned memory for async transfers
        cudaMallocHost(&h_in, bytes);
        cudaMallocHost(&h_out, bytes);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        // Initialize
        for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        constexpr int block_size = 128;
        int threads = block_size;
        // Shared memory: 2 stages * threads * sizeof(float)
        size_t smem_size = 2 * threads * sizeof(float);

        printf("Launching kernel with %d threads, %zu bytes shared mem\n", threads, smem_size);

        // Launch (1 block for demo simplicity)
        async_copy_kernel<block_size><<<1, threads, smem_size>>>(d_out, d_in, N, compute_load);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel failed: %s\n", cudaGetErrorString(err));
        } else {
            printf("Kernel completed successfully.\n");
        }

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    }
};
