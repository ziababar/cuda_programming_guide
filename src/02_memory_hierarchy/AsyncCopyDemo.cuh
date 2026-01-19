#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

// Compile with -arch=sm_80 or higher
// This kernel demonstrates a simple pipeline using cp.async (via cuda::pipeline)
__global__ void async_pipeline_kernel(int* global_data, int* output_data, int N) {
    extern __shared__ int s_data[];

    // Create a pipeline object with thread scope
    // This allows coordinating async copies at the thread level
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        // Stage 1: Issue async copy from global to shared
        // The memcpy_async is equivalent to cp.async instructions
        pipe.producer_acquire();
        cuda::memcpy_async(&s_data[tid], &global_data[idx], sizeof(int), pipe);
        pipe.producer_commit();

        // Stage 2: Wait for copy to complete
        // consumer_wait() blocks until the specified stages are complete
        pipe.consumer_wait();

        // Stage 3: Compute using data in shared memory
        // At this point, data is guaranteed to be in s_data
        int val = s_data[tid];
        val *= 2; // Simple computation

        // Release stage (cleanup)
        pipe.consumer_release();

        // Write back result to global memory
        output_data[idx] = val;
    }
}

class AsyncCopyDemo {
public:
    static void run() {
        printf("=== Asynchronous Copy (cp.async) Demo ===\n");

        int dev_id = 0;
        cudaError_t err = cudaGetDevice(&dev_id);
        if (err != cudaSuccess) {
            printf("Error getting device: %s\n", cudaGetErrorString(err));
            return;
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev_id);

        // Check for Ampere (Compute Capability 8.0) or higher
        if (prop.major < 8) {
            printf("Skipping: Async Copy requires Compute Capability 8.0+ (Ampere).\n");
            printf("Current device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
            return;
        }

        const int N = 1024;
        const int bytes = N * sizeof(int);

        // Allocate host memory
        // For async operations, pinned memory is often preferred, but not strictly required for cp.async (which is device-side)
        int* h_in;
        int* h_out;
        cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_out, bytes, cudaHostAllocDefault);

        for (int i = 0; i < N; i++) h_in[i] = i;

        int *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        int shared_mem_size = threads * sizeof(int);

        printf("Launching kernel with %d blocks, %d threads\n", blocks, threads);
        async_pipeline_kernel<<<blocks, threads, shared_mem_size>>>(d_in, d_out, N);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
             printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for (int i = 0; i < N; i++) {
            if (h_out[i] != h_in[i] * 2) {
                printf("Error at %d: Expected %d, got %d\n", i, h_in[i]*2, h_out[i]);
                correct = false;
                break;
            }
        }

        if (correct) printf("Success! Async copy kernel verified.\n");

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
