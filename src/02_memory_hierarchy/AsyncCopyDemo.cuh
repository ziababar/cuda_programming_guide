#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

// Compile with -arch=sm_80 or higher to enable cp.async

// Kernel demonstrating a simple async copy pipeline using cuda::pipeline
// This kernel copies data from global memory to shared memory asynchronously,
// performs a dummy computation, and writes back.
// Ideally, this would be a multi-stage pipeline, but for simplicity, we show the mechanism.
__global__ void async_copy_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    // Shared memory for double buffering (simplistic view here)
    extern __shared__ float shared_mem[];
    float* buffer = shared_mem;

    auto block = cg::this_thread_block();
    int tid = block.thread_rank();
    int gmem_idx = block.group_index().x * block.size() + tid;

    if (gmem_idx >= N) return;

    // Create a pipeline object
    // thread_scope_thread means the pipeline is managed by this individual thread.
    // For block-wide cooperation, you would use thread_scope_block.
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // 1. Acquire: Reserve resources for the copy
    pipe.producer_acquire();

    // 2. Submit: Asynchronous copy from Global -> Shared
    //    Note: This bypasses the register file.
    cuda::memcpy_async(&buffer[tid], &input[gmem_idx], sizeof(float), pipe);

    // 3. Commit: Mark the end of the batch of copy operations
    pipe.producer_commit();

    // ... In a real pipeline, we would compute on the *previous* batch here ...

    // 4. Wait: Wait for the copy to complete
    pipe.consumer_wait();

    // 5. Compute: Now safe to read from shared memory
    float val = buffer[tid];
    val = val * 2.0f; // Dummy computation

    // 6. Release: Signal we are done reading the buffer
    pipe.consumer_release();

    // Store result
    output[gmem_idx] = val;
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Memory Copy (cp.async) Demo ===\n");

        // Check for Compute Capability 8.0+
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (prop.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere). Detected %d.%d\n", prop.major, prop.minor);
            return;
        }

        const int N = 1024 * 1024;
        const int bytes = N * sizeof(float);

        // Host memory
        std::vector<float> h_input(N);
        std::vector<float> h_output(N);

        for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

        // Device memory
        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);

        cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

        // Launch settings
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        // Allocate shared memory: 1 float per thread
        size_t shared_mem_size = threads * sizeof(float);

        async_copy_kernel<<<blocks, threads, shared_mem_size>>>(d_input, d_output, N);
        cudaDeviceSynchronize();

        // Check errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        } else {
            cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
            // Verify
            bool correct = true;
            for (int i = 0; i < N; ++i) {
                if (h_output[i] != 2.0f) {
                    printf("Mismatch at %d: %f != 2.0f\n", i, h_output[i]);
                    correct = false;
                    break;
                }
            }
            if (correct) printf("Verification Successful: All values doubled.\n");
        }

        cudaFree(d_input);
        cudaFree(d_output);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
