#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda/pipeline>
#include <cstdio>
#include <vector>

// Helper macro to check for Ampere+ architecture
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // On older architectures, fallback or error
    #define ASYNC_COPY_SUPPORTED 0
#else
    #define ASYNC_COPY_SUPPORTED 1
#endif

// Kernel demonstrating Async Copy using cuda::pipeline
// Requires Compute Capability 8.0+ (Ampere)
__global__ void async_copy_pipeline_kernel(const int* __restrict__ global_input,
                                           int* global_output,
                                           int N) {
#if ASYNC_COPY_SUPPORTED
    extern __shared__ int shared_mem[];
    int* s_buff = shared_mem;

    // Create a pipeline object for this thread
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Calculate global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Phase 1: Issue Async Copy
        // Acquire a stage in the pipeline
        pipe.producer_acquire();

        // Copy from Global to Shared asynchronously
        // Note: Using &s_buff[threadIdx.x] as destination
        cuda::memcpy_async(&s_buff[threadIdx.x],
                           &global_input[idx],
                           sizeof(int),
                           pipe);

        // Commit the stage
        pipe.producer_commit();

        // In a real app, we could do independent work here while memory loads
        // ... independent_compute() ...

        // Phase 2: Wait for Copy to Complete
        // Wait for all stages to complete
        pipe.consumer_wait();

        // Phase 3: Read from Shared Memory and Compute
        // Now it's safe to read s_buff
        int val = s_buff[threadIdx.x];

        // Simple computation: multiply by 2
        val *= 2;

        // Store result back to global memory
        global_output[idx] = val;

        // Release the stage
        pipe.consumer_release();
    }
#else
    // Fallback for older architectures
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        global_output[idx] = global_input[idx] * 2;
    }
#endif
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Memory Copy (cp.async) Demo ===\n");

        int device_id = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        if (prop.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere). Detected SM %d.%d\n",
                   prop.major, prop.minor);
            return;
        }

        const int N = 256;
        size_t bytes = N * sizeof(int);

        // Host memory
        std::vector<int> h_in(N);
        std::vector<int> h_out(N);
        for (int i = 0; i < N; i++) h_in[i] = i;

        // Device memory
        int *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

        // Launch kernel
        int threads = 128;
        int blocks = (N + threads - 1) / threads;
        size_t shared_mem_size = threads * sizeof(int);

        printf("Launching kernel with %d blocks, %d threads, %zu bytes shared mem\n",
               blocks, threads, shared_mem_size);

        async_copy_pipeline_kernel<<<blocks, threads, shared_mem_size>>>(d_in, d_out, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();

        // Verify results
        cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
        bool correct = true;
        for (int i = 0; i < N; i++) {
            if (h_out[i] != h_in[i] * 2) {
                printf("Mismatch at %d: Expected %d, Got %d\n", i, h_in[i] * 2, h_out[i]);
                correct = false;
                break;
            }
        }

        if (correct) {
            printf("SUCCESS: Async copy kernel results verified.\n");
        } else {
            printf("FAILURE: Results mismatch.\n");
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
