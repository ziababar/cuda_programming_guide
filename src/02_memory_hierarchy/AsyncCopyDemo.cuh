#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

// Kernel demonstrating cp.async using cuda::pipeline
// Requires Compute Capability 8.0+ (Ampere or later)
__global__ void async_copy_pipeline_kernel(float* d_out, const float* d_in, int N) {
    // Shared memory buffer
    extern __shared__ float s_data[];

    // Create a pipeline object
    // cuda::thread_scope_thread: The pipeline is managed by this thread
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        // --- Stage 1: Issue Asynchronous Copy ---

        // Acquire the pipeline for producing (issuing copies)
        pipe.producer_acquire();

        // Issue asynchronous copy from global to shared memory
        // cuda::memcpy_async(dest, src, size, pipeline)
        // This maps to the cp.async PTX instruction
        cuda::memcpy_async(&s_data[tid], &d_in[idx], sizeof(float), pipe);

        // Commit the issued commands
        pipe.producer_commit();

        // --- Stage 2: Wait for Data ---

        // Wait for the copy to complete
        pipe.consumer_wait();

        // --- Stage 3: Compute ---

        // Access data from shared memory (now guaranteed to be ready)
        float val = s_data[tid];
        val = val * 2.0f;

        // Release the pipeline resource
        pipe.consumer_release();

        // Write result back to global memory
        d_out[idx] = val;
    }
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Data Copy (cp.async) Demo ===\n");

        // Check Compute Capability
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (prop.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere). Current: %d.%d\n", prop.major, prop.minor);
            return;
        }

        const int N = 1024;
        size_t bytes = N * sizeof(float);

        // Host data
        std::vector<float> h_in(N, 1.0f);
        std::vector<float> h_out(N, 0.0f);

        // Device data
        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

        // Launch kernel
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        size_t shared_mem_size = threads * sizeof(float);

        printf("Launching kernel with %d blocks, %d threads, %zu bytes shared mem...\n", blocks, threads, shared_mem_size);
        async_copy_pipeline_kernel<<<blocks, threads, shared_mem_size>>>(d_out, d_in, N);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();

        // Verify
        cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for (int i = 0; i < N; ++i) {
            if (h_out[i] != 2.0f) {
                printf("Mismatch at %d: Expected 2.0, got %f\n", i, h_out[i]);
                correct = false;
                break;
            }
        }

        if (correct) {
            printf("Success: All values doubled correctly using cp.async pipeline.\n");
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
