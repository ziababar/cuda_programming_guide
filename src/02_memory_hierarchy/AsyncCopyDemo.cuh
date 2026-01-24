#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

// Kernel demonstrating Asynchronous Memory Copy using cuda::pipeline
// This kernel copies data from Global to Shared memory asynchronously.
// Requires Compute Capability 8.0+ (Ampere)
template <int BLOCK_SIZE>
__global__ void async_copy_pipeline_kernel(int* global_data, int* output_data, size_t n) {
    extern __shared__ int shared_buffer[]; // Dynamic shared memory
    int* s_data = shared_buffer;

    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_idx = threadIdx.x;

    // Create a pipeline object with thread scope
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // 1. ASYNC COPY: Initiate copy from Global to Shared
    // The copy is issued but not necessarily completed
    pipe.producer_acquire();
    if (global_idx < n) {
        // cp.async under the hood
        cuda::memcpy_async(&s_data[local_idx], &global_data[global_idx], sizeof(int), pipe);
    }
    pipe.producer_commit();

    // 2. COMPUTE INDEPENDENT WORK
    // We can do work here that doesn't depend on the data being copied
    // (In this simple example, we don't have much independent work)

    // 3. WAIT: Wait for the copy to complete
    pipe.consumer_wait();

    // 4. CONSUME: Use the data in Shared Memory
    if (global_idx < n) {
        // Perform some computation (e.g., simple scaling)
        int val = s_data[local_idx];
        val = val * 2;
        output_data[global_idx] = val;
    }

    // Release the pipeline stage (cleanup)
    pipe.consumer_release();
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Copy (cp.async) Demo ===\n");

        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        if (prop.major < 8) {
            printf("Skipping: Async Copy requires Compute Capability 8.0+ (Ampere)\n");
            return;
        }

        const size_t N = 1024 * 1024;
        const size_t bytes = N * sizeof(int);
        const int threads_per_block = 256;
        const int blocks = (N + threads_per_block - 1) / threads_per_block;

        // Host Memory (Pinned is recommended for best async performance)
        int *h_data, *h_output;
        cudaHostAlloc(&h_data, bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_output, bytes, cudaHostAllocDefault);

        for (size_t i = 0; i < N; ++i) h_data[i] = 1;

        // Device Memory
        int *d_data, *d_output;
        cudaMalloc(&d_data, bytes);
        cudaMalloc(&d_output, bytes);

        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

        // Launch Kernel
        // Shared memory size needed: threads_per_block * sizeof(int)
        size_t shmem_size = threads_per_block * sizeof(int);
        async_copy_pipeline_kernel<threads_per_block><<<blocks, threads_per_block, shmem_size>>>(d_data, d_output, N);
        cudaDeviceSynchronize();

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        } else {
            cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
            // Verify
            bool correct = true;
            for (size_t i = 0; i < N; ++i) {
                if (h_output[i] != 2) {
                    correct = false;
                    printf("Mismatch at %zu: %d != 2\n", i, h_output[i]);
                    break;
                }
            }
            if (correct) printf("Success: All values doubled correctly via Async Copy pipeline.\n");
        }

        cudaFree(d_data);
        cudaFree(d_output);
        cudaFreeHost(h_data);
        cudaFreeHost(h_output);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
