#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath> // For std::abs
#include <cuda/pipeline>

// Requires Compute Capability 8.0+ (Ampere)

// Kernel using cuda::pipeline for async global-to-shared copy with Double Buffering
// This demonstrates true latency hiding: overlapping memory transfer with computation.
__global__ void async_copy_pipeline_kernel(const float* __restrict__ global_in,
                                          float* __restrict__ global_out,
                                          int N) {
    // Shared memory for 2 stages (Double Buffering)
    extern __shared__ float shared_mem[];
    float* s_buff[2];
    s_buff[0] = shared_mem;
    s_buff[1] = shared_mem + blockDim.x;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int batch_size = blockDim.x;

    // Prologue: Issue copy for the first batch (Batch 0) into buffer 0
    int i = 0;
    pipe.producer_acquire();
    if (i + tid < N) {
        cuda::memcpy_async(&s_buff[0][tid], &global_in[i + tid], sizeof(float), pipe);
    }
    pipe.producer_commit();

    // Main loop
    for (; i < N; i += batch_size) {
        int stage = (i / batch_size) % 2;       // Current stage buffer index
        int next_stage = (stage + 1) % 2;       // Next stage buffer index
        int next_i = i + batch_size;

        // Issue copy for NEXT batch into next buffer (if valid)
        if (next_i < N) {
            pipe.producer_acquire();
            if (next_i + tid < N) {
                cuda::memcpy_async(&s_buff[next_stage][tid], &global_in[next_i + tid], sizeof(float), pipe);
            }
            pipe.producer_commit();
        }

        // Wait for CURRENT batch copy to complete
        pipe.consumer_wait();

        // Compute CURRENT batch (from current buffer)
        // While we compute here, the hardware is fetching the next batch in the background!
        if (i + tid < N) {
            float val = s_buff[stage][tid];
            val = val * val; // Simple computation
            global_out[i + tid] = val;
        }

        // Release CURRENT batch stage
        pipe.consumer_release();
    }
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Async Memory Copy (cp.async) Demo ===\n");

        int dev = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        if (prop.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere or newer).\n");
            printf("Current device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
            return;
        }

        printf("Running on %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);

        const int N = 1024 * 1024;
        const int bytes = N * sizeof(float);
        const int block_size = 256;
        const int grid_size = (N + block_size - 1) / block_size;

        // Allocate host memory (pinned for best performance)
        float *h_in, *h_out;
        cudaMallocHost(&h_in, bytes);
        cudaMallocHost(&h_out, bytes);

        // Initialize with smaller values to avoid precision issues
        for (int i = 0; i < N; ++i) {
            h_in[i] = (float)(i % 100);
        }

        // Allocate device memory
        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        // Launch kernel
        // Shared memory size = 2 * block_size * sizeof(float) for double buffering
        size_t shared_mem_size = 2 * block_size * sizeof(float);

        printf("Launching kernel with Double Buffering Pipeline...\n");
        async_copy_pipeline_kernel<<<grid_size, block_size, shared_mem_size>>>(d_in, d_out, N);
        cudaDeviceSynchronize();

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        // Verify results
        bool correct = true;
        for (int i = 0; i < N; ++i) {
            float expected = h_in[i] * h_in[i];
            if (std::abs(h_out[i] - expected) > 1e-4) {
                if (i < 10) printf("Mismatch at %d: %f != %f\n", i, h_out[i], expected);
                correct = false;
                break;
            }
        }

        if (correct) {
            printf("Validation Successful!\n");
        } else {
            printf("Validation Failed!\n");
        }

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
