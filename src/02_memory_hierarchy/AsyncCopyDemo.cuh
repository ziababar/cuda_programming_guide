#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cstdio>
#include <vector>

// Required for cp.async
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda/pipeline>
#endif

// Kernel demonstrating Async Copy with double buffering
// This kernel computes a simple transformation on data loaded via cp.async
// It uses a 2-stage pipeline (double buffering)
__global__ void async_copy_kernel(const float* __restrict__ global_in,
                                  float* __restrict__ global_out,
                                  int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ float shared_mem[];

    // Divide shared memory into two buffers for double buffering
    // Each buffer holds blockDim.x elements
    float* buffer0 = shared_mem;
    float* buffer1 = shared_mem + blockDim.x;

    // Pipeline object
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Number of batches to process
    // We process blockDim.x elements per batch per block
    // Simplification: Assume N is a multiple of blockDim.x * gridDim.x
    // Each block processes a chunk of size (N / gridDim.x)
    int elements_per_block = N / gridDim.x;
    int batches_per_block = elements_per_block / blockDim.x;

    // Global offset for this block
    int block_offset = blockIdx.x * elements_per_block;

    // Prologue: Load the first batch into buffer0
    pipe.producer_acquire();
    cuda::memcpy_async(&buffer0[threadIdx.x],
                       &global_in[block_offset + threadIdx.x],
                       sizeof(float), pipe);
    pipe.producer_commit();

    // Loop over batches
    for (int i = 0; i < batches_per_block; ++i) {
        // Determine current and next buffer
        float* current_buffer = (i % 2 == 0) ? buffer0 : buffer1;
        float* next_buffer    = (i % 2 == 0) ? buffer1 : buffer0;

        // Issue fetch for next batch (if not the last one)
        if (i < batches_per_block - 1) {
            int next_batch_idx = i + 1;
            int global_idx = block_offset + next_batch_idx * blockDim.x + threadIdx.x;

            pipe.producer_acquire();
            cuda::memcpy_async(&next_buffer[threadIdx.x],
                               &global_in[global_idx],
                               sizeof(float), pipe);
            pipe.producer_commit();
        }

        // Wait for current batch to be ready
        pipe.consumer_wait();

        // COMPUTE: Process the current batch
        float val = current_buffer[threadIdx.x];
        val = val * 2.0f + 1.0f; // Simple compute

        // Write result to global memory (standard store)
        // Note: For maximum performance, async store could also be used if supported/needed
        int output_idx = block_offset + i * blockDim.x + threadIdx.x;
        global_out[output_idx] = val;

        // Release the buffer (we are done reading from it)
        pipe.consumer_release();
    }
#else
    // Fallback for older architectures
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        global_out[tid] = global_in[tid] * 2.0f + 1.0f;
    }
#endif
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Data Copy (cp.async) Demo ===\n");

        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        if (prop.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere). Detected %d.%d\n", prop.major, prop.minor);
            return;
        }

        const int N = 1024 * 1024;
        const int bytes = N * sizeof(float);

        // Host memory
        std::vector<float> h_in(N);
        std::vector<float> h_out(N);
        for(int i=0; i<N; ++i) h_in[i] = 1.0f;

        // Device memory
        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

        // Launch configuration
        int threads = 128;
        int blocks = N / threads; // Assume perfect division

        // Shared memory: 2 buffers * threads * sizeof(float)
        size_t shared_mem_size = 2 * threads * sizeof(float);

        printf("Launching kernel with %d blocks, %d threads, %lu bytes shared mem\n", blocks, threads, shared_mem_size);
        async_copy_kernel<<<blocks, threads, shared_mem_size>>>(d_in, d_out, N);

        cudaDeviceSynchronize();

        // Verify
        cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
        bool correct = true;
        for(int i=0; i<N; ++i) {
            if (abs(h_out[i] - 3.0f) > 1e-5) {
                correct = false;
                printf("Error at %d: Expected 3.0, got %f\n", i, h_out[i]);
                break;
            }
        }

        if (correct) {
            printf("Success! All values computed correctly using async copy pipeline.\n");
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
