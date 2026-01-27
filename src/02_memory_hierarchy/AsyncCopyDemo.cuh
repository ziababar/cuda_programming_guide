#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cstdio>
#include <vector>

// Compile with -arch=sm_80 or higher
// cuda::pipeline requires CUDA 11.0+ and Compute Capability 8.0+

__global__ void async_copy_pipeline_kernel(float* d_out, const float* d_in, int N) {
    // Shared memory for the tile
    extern __shared__ float shared_mem[];
    float* tile = shared_mem;

    // Create a pipeline object for this thread
    // cuda::thread_scope_thread means the pipeline is managed by this thread
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    if (global_idx < N) {
        // 1. ACQUIRE: Reserve space in the pipeline
        pipe.producer_acquire();

        // 2. COPY: Issue asynchronous copy from Global to Shared Memory
        // This bypasses the register file, saving register pressure
        cuda::memcpy_async(&tile[tid], &d_in[global_idx], sizeof(float), pipe);

        // 3. COMMIT: Commit the copy command
        pipe.producer_commit();

        // --- Overlap Opportunity ---
        // We could do independent math here while memory is fetching!
        // ---------------------------

        // 4. WAIT: Wait for the copy to finish
        // consumer_wait() blocks until the specified batch is done
        pipe.consumer_wait();

        // 5. CONSUME: Use the data from shared memory
        float val = tile[tid];
        d_out[global_idx] = val * 2.0f;

        // 6. RELEASE: Release the stage
        pipe.consumer_release();
    }
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Memory Copy (cp.async) Demo ===\n");

        int dev_id = 0;
        cudaGetDevice(&dev_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev_id);

        if (prop.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere). Detected %d.%d\n", prop.major, prop.minor);
            return;
        }

        const int N = 1024;
        const int bytes = N * sizeof(float);

        // Host memory
        std::vector<float> h_in(N);
        std::vector<float> h_out(N);
        for (int i = 0; i < N; ++i) h_in[i] = 1.0f * i;

        // Device memory
        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        // Use pinned memory for better async performance in real apps,
        // but here we just copy initially
        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

        // Launch kernel
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        int shared_mem_size = threads * sizeof(float);

        printf("Launching kernel with %d blocks, %d threads, %d bytes shared mem...\n", blocks, threads, shared_mem_size);
        async_copy_pipeline_kernel<<<blocks, threads, shared_mem_size>>>(d_out, d_in, N);
        cudaDeviceSynchronize();

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        } else {
            cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

            // Verify
            bool correct = true;
            for (int i = 0; i < N; ++i) {
                if (abs(h_out[i] - (h_in[i] * 2.0f)) > 1e-5) {
                    correct = false;
                    printf("Mismatch at %d: Expected %f, Got %f\n", i, h_in[i] * 2.0f, h_out[i]);
                    break;
                }
            }
            if (correct) {
                printf("Verification Successful!\n");
            }
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
