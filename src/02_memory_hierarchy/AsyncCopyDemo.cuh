#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cstdio>
#include <vector>

// Modern CUDA Asynchronous Copy Demo (Compute Capability 8.0+)
// Uses cuda::pipeline to manage asynchronous data movement from Global to Shared Memory.

const int BLOCK_SIZE = 128;
const int TILE_SIZE = 1024; // Elements per tile

__global__ void async_copy_kernel(int* d_out, const int* d_in, int N) {
    extern __shared__ int smem[];
    int* s_data = smem;

    // Create a pipeline object for the current thread
    // cuda::thread_scope_thread means the pipeline state is local to the thread
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Loop over tiles
    for (int i = 0; i < N; i += TILE_SIZE) {
        // Acquire a stage in the pipeline
        pipe.producer_acquire();

        // Each thread copies a portion of the tile
        int elements_remaining = N - i;
        int elements_to_copy = (elements_remaining < TILE_SIZE) ? elements_remaining : TILE_SIZE;

        for (int j = threadIdx.x; j < elements_to_copy; j += blockDim.x) {
            // Asynchronous copy from global to shared memory
            // This bypasses the register file, saving register pressure and hiding latency
            cuda::memcpy_async(&s_data[j], &d_in[i + j], sizeof(int), pipe);
        }

        // Commit the operations for this stage
        pipe.producer_commit();

        // Wait for the copy to complete before using the data
        pipe.consumer_wait();

        // Sync threads to ensure all data is visible in shared memory
        __syncthreads();

        // Compute: Simply double the value and write back to global memory
        // (In a real app, you would do heavy compute here while prefetching the next tile)
        for (int j = threadIdx.x; j < elements_to_copy; j += blockDim.x) {
            d_out[i + j] = s_data[j] * 2;
        }

        // Release the stage
        pipe.consumer_release();

        // Sync before starting next iteration to protect shared memory
        __syncthreads();
    }
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Memory Copy (cp.async) Demo ===\n");

        // Check Compute Capability
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (prop.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere). Detected %d.%d\n", prop.major, prop.minor);
            return;
        }

        const int N = 1024 * 1024;
        const int bytes = N * sizeof(int);

        // Host data
        std::vector<int> h_in(N);
        std::vector<int> h_out(N);
        for (int i = 0; i < N; ++i) h_in[i] = i;

        // Device data
        int *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

        // Launch kernel
        // Shared memory size needed: TILE_SIZE * sizeof(int)
        async_copy_kernel<<<N / TILE_SIZE, BLOCK_SIZE, TILE_SIZE * sizeof(int)>>>(d_out, d_in, N);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        } else {
            cudaDeviceSynchronize();

            // Verify
            cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
            bool correct = true;
            for (int i = 0; i < N; ++i) {
                if (h_out[i] != h_in[i] * 2) {
                    printf("Mismatch at %d: Expected %d, Got %d\n", i, h_in[i] * 2, h_out[i]);
                    correct = false;
                    break;
                }
            }
            if (correct) {
                printf("Async copy and compute successful!\n");
            }
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
