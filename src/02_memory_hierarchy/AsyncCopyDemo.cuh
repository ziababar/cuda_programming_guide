#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

// Modern Async Copy Demo using cuda::pipeline
// Requires Compute Capability 8.0+ (Ampere)
// Compile with: -arch=sm_80

constexpr int TILE_SIZE = 128; // Threads per block
constexpr int STAGES = 4;      // Number of pipeline stages

// Implementation of a multi-stage async copy pipeline
// This kernel processes 'num_batches' of TILE_SIZE elements per block.
__global__ void async_copy_pipeline_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int num_batches) {
    // Shared memory for 4-stage pipeline (circular buffer)
    extern __shared__ float shared_buffer[];
    float* smem_stages[STAGES];
    for (int i = 0; i < STAGES; ++i) {
        smem_stages[i] = shared_buffer + i * TILE_SIZE;
    }

    // Pipeline object with thread scope.
    // cuda::thread_scope_thread means each thread manages its own copy/wait state.
    // For collaborative loading (e.g. GEMM), use cuda::thread_scope_block.
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Prologue: Prime the pipeline
    // Load the first (STAGES - 1) batches to fill the pipeline latency
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_batches) {
            int src_idx = tid + s * stride;
            pipe.producer_acquire();
            // Initiate async copy from Global to Shared
            // Bypasses register file
            cuda::memcpy_async(&smem_stages[s][threadIdx.x],
                               &input[src_idx],
                               sizeof(float),
                               pipe);
            pipe.producer_commit();
        }
    }

    // Main Loop: Compute and Fetch Next
    for (int batch = 0; batch < num_batches; ++batch) {
        // 1. Issue load for the future batch (batch + STAGES - 1)
        int next_batch = batch + STAGES - 1;
        if (next_batch < num_batches) {
            int stage_idx = next_batch % STAGES;
            int src_idx = tid + next_batch * stride;

            pipe.producer_acquire();
            cuda::memcpy_async(&smem_stages[stage_idx][threadIdx.x],
                               &input[src_idx],
                               sizeof(float),
                               pipe);
            pipe.producer_commit();
        }

        // 2. Wait for the current batch's data to be ready
        // Blocks until the stage we need is complete
        pipe.consumer_wait();

        // 3. Compute on the data from Shared Memory
        int curr_stage = batch % STAGES;
        float val = smem_stages[curr_stage][threadIdx.x];

        // Simple computation (square + 1)
        val = val * val + 1.0f;

        // 4. Store result (Synchronous global store)
        int dst_idx = tid + batch * stride;
        output[dst_idx] = val;

        // 5. Release the stage buffer so it can be reused
        pipe.consumer_release();
    }
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Async Copy Pipeline Demo ===\n");
        printf("Note: Requires Compute Capability 8.0+ (Ampere)\n");

        int batch_count = 100;
        int threads = TILE_SIZE; // 128
        int blocks = 1;          // Single block for demonstration clarity
        int N = batch_count * blocks * threads;

        size_t bytes = N * sizeof(float);

        float *h_in, *h_out;
        cudaMallocHost(&h_in, bytes);
        cudaMallocHost(&h_out, bytes);

        for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        // Shared memory size: STAGES * TILE_SIZE * sizeof(float)
        size_t smem_size = STAGES * threads * sizeof(float);

        async_copy_pipeline_kernel<<<blocks, threads, smem_size>>>(d_in, d_out, batch_count);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        } else {
            cudaDeviceSynchronize();

            cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

            // Verify results
            bool passed = true;
            if (h_out[0] != 2.0f) passed = false;
            if (h_out[N-1] != 2.0f) passed = false;

            if (passed) {
                printf("Verification Passed: First=%f, Last=%f\n", h_out[0], h_out[N-1]);
            } else {
                printf("Verification Failed: First=%f, Last=%f (Expected 2.0)\n", h_out[0], h_out[N-1]);
            }
        }

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
