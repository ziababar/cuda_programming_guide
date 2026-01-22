#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

// Kernel demonstrating Async Copy (cp.async) using cuda::pipeline
// Requires Compute Capability 8.0+ (Ampere)
template <int BLOCK_SIZE>
__global__ void async_copy_pipeline_kernel(int* global_in, int* global_out, int N) {
#if __CUDA_ARCH__ >= 800
    // Shared memory buffer
    extern __shared__ int shared_mem[];
    int* s_mem = shared_mem;

    // Create a pipeline object with thread scope
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Loop over the data in tiles
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int num_batches = N / BLOCK_SIZE;

    // Prologue: Start loading the first batch
    if (idx < N) {
        pipe.producer_acquire();
        cuda::memcpy_async(s_mem + tid, global_in + idx, sizeof(int), pipe);
        pipe.producer_commit();
    }

    // Main loop
    for (int i = 0; i < num_batches; ++i) {
        // Wait for the current batch (stage) to finish loading
        // consumer_wait() blocks until the oldest committed stage is complete
        pipe.consumer_wait();

        // --- Compute Phase ---
        int val = s_mem[tid];
        val = val * 2;

        // Write back (synchronous)
        global_out[idx] = val;

        // Release the pipeline stage we just consumed
        pipe.consumer_release();

        idx += BLOCK_SIZE;

        // Note: For a single buffer, we serialize Load -> Wait -> Compute.
        // True overlap requires double buffering (see below).
    }
#else
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        global_out[tid] = global_in[tid] * 2;
    }
#endif
}

// Double-buffered Pipeline Example
// This demonstrates true overlap of Copy and Compute
template <int BLOCK_SIZE>
__global__ void double_buffered_pipeline_kernel(int* global_in, int* global_out, int N) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ int shared_mem[];
    // Split shared memory into two buffers
    int* s_buff[2];
    s_buff[0] = shared_mem;
    s_buff[1] = shared_mem + BLOCK_SIZE;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int total_batches = N / BLOCK_SIZE;

    // Prologue: Load Batch 0 into Buffer 0
    pipe.producer_acquire();
    cuda::memcpy_async(s_buff[0] + tid, global_in + global_idx, sizeof(int), pipe);
    pipe.producer_commit();

    // Steady State Loop
    for (int i = 0; i < total_batches; ++i) {
        int curr_buff = i % 2;
        int next_buff = (i + 1) % 2;

        // 1. Issue Load for Batch i+1 (if exists) into next_buff
        // This runs concurrently with the compute on Batch i (after wait)
        // Note: We issue BEFORE waiting to maximize overlap window if we had independent work,
        // but here we need to acquire the stage first.

        if (i < total_batches - 1) {
            int next_global_idx = global_idx + BLOCK_SIZE;
            pipe.producer_acquire(); // Reserve resources for next stage
            cuda::memcpy_async(s_buff[next_buff] + tid, global_in + next_global_idx, sizeof(int), pipe);
            pipe.producer_commit(); // Commit Batch i+1
        }

        // 2. Wait for Batch i to complete
        // consumer_wait() waits for the stage at the consumer cursor (oldest active stage)
        pipe.consumer_wait();

        // 3. Compute on Batch i (in curr_buff)
        int val = s_buff[curr_buff][tid];
        val = val * 2;
        global_out[global_idx] = val;

        // 4. Release Batch i
        // This makes the stage available for reuse by the producer
        pipe.consumer_release();

        global_idx += BLOCK_SIZE;
    }
#endif
}

class AsyncCopyDemo {
public:
    static void run_demos() {
        printf("=== Async Copy (cp.async) Demo ===\n");

        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        if (prop.major < 8) {
            printf("Skipping Async Copy demo: Requires Compute Capability 8.0+ (Ampere). Detected: %d.%d\n", prop.major, prop.minor);
            return;
        }

        const int N = 1024 * 1024;
        const int BLOCK_SIZE = 256;
        size_t bytes = N * sizeof(int);

        // Host memory
        int *h_in, *h_out;
        cudaMallocHost(&h_in, bytes);
        cudaMallocHost(&h_out, bytes);

        for (int i = 0; i < N; ++i) h_in[i] = 1;

        int *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        // Run Kernel
        printf("Running Double Buffered Pipeline Kernel...\n");
        // Shared mem size = 2 * BLOCK_SIZE * sizeof(int)
        double_buffered_pipeline_kernel<BLOCK_SIZE><<<N / BLOCK_SIZE, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);

        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        // Verify
        bool correct = true;
        for (int i = 0; i < 10; ++i) {
            if (h_out[i] != 2) {
                correct = false;
                printf("Mismatch at %d: %d != 2\n", i, h_out[i]);
                break;
            }
        }
        if (correct) printf("Verification Passed!\n");

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
