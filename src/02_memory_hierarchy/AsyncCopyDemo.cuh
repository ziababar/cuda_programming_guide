#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cstdio>
#include <vector>

// Check for Compute Capability 8.0+ (Ampere)
#if __CUDA_ARCH__ >= 800
    #define ASYNC_COPY_SUPPORTED 1
#endif

// Kernel demonstrating Async Copy using cuda::pipeline
// Each block processes a chunk of data.
// We use a pipeline to overlap loading the NEXT chunk while processing the CURRENT chunk.
template <int BLOCK_SIZE>
__global__ void async_copy_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
#ifdef ASYNC_COPY_SUPPORTED
    // Shared memory for double buffering
    // 2 stages: 0 for processing, 1 for loading
    extern __shared__ float shared_mem[];
    float* stage0 = &shared_mem[0];
    float* stage1 = &shared_mem[BLOCK_SIZE];

    // Initialize pipeline
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * blockDim.x;

    // Loop over the data
    for (size_t i = idx; i < n; i += grid_stride) {
        // For simplicity in this demo, we won't implement a full multi-stage loop with prologue/epilogue
        // to keep it readable. Instead, we'll demonstrate a single stage load-wait-compute.
        // A real optimized kernel would use multi-stage buffering.

        // 1. Submit Async Copy: Global -> Shared
        pipe.producer_acquire();
        cuda::memcpy_async(&stage0[threadIdx.x], &input[i], sizeof(float), pipe);
        pipe.producer_commit();

        // 2. Wait for copy to complete
        pipe.consumer_wait();

        // 3. Compute (simple square)
        float val = stage0[threadIdx.x];
        val = val * val;

        // 4. Store result
        output[i] = val;

        // Release the stage
        pipe.consumer_release();
    }
#else
    // Fallback for older architectures
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = val * val;
    }
#endif
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Memory Copy (cp.async) Demo ===\n");

        // Check device capability
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (prop.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere). Detected %d.%d\n", prop.major, prop.minor);
            return;
        }

        const int N = 1024 * 1024;
        const int BLOCK_SIZE = 256;
        size_t bytes = N * sizeof(float);

        // Host memory
        // Ideally use cudaHostAlloc for true async overlap potential on host side,
        // but here we focus on device-side async copy.
        std::vector<float> h_in(N);
        std::vector<float> h_out(N);

        for (int i = 0; i < N; ++i) h_in[i] = (float)i;

        // Device memory
        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

        // Launch kernel
        // Shared memory size: 2 buffers * BLOCK_SIZE * sizeof(float)
        size_t shared_mem_size = 2 * BLOCK_SIZE * sizeof(float);

        async_copy_kernel<BLOCK_SIZE><<< (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_mem_size >>>(d_in, d_out, N);

        cudaDeviceSynchronize();

        // Check errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        } else {
            printf("Kernel execution successful.\n");
        }

        // Verify result (partial)
        cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
        bool match = true;
        for (int i = 0; i < 10; ++i) { // Check first 10
             if (abs(h_out[i] - (h_in[i] * h_in[i])) > 1e-5) {
                 match = false;
                 printf("Mismatch at %d: Expected %f, got %f\n", i, h_in[i] * h_in[i], h_out[i]);
                 break;
             }
        }

        if (match) printf("Verification passed.\n");

        cudaFree(d_in);
        cudaFree(d_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
