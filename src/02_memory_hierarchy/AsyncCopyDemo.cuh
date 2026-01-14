#ifndef ASYNC_COPY_DEMO_CUH
#define ASYNC_COPY_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cstdio>
#include <vector>

// Requires Compute Capability 8.0+ (Ampere)
// Compile with -arch=sm_80

// Asynchronous Copy using cuda::pipeline
// Loads data from Global to Shared Memory asynchronously
__global__ void async_copy_kernel(const int* __restrict__ global_data, int* results, int N) {
    extern __shared__ int shared_buffer[];

    // Create a pipeline object
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Use threadIdx.x for indexing within shared memory
    int tx = threadIdx.x;

    // Start the pipeline
    // Simple pipelining strategy:
    // 1. Issue copy for current batch
    // 2. Wait for it to complete
    // 3. Compute

    for (int i = 0; i < N; i += blockDim.x) {
        int idx = i + tx;

        // Active threads issue the copy
        if (idx < N) {
            // Issue async copy from global to shared
            pipe.producer_acquire();
            cuda::memcpy_async(&shared_buffer[tx], &global_data[idx], sizeof(int), pipe);
            pipe.producer_commit();

            // Wait for the copy to complete
            pipe.consumer_wait();

            // Consume data from shared memory (Compute stage)
            results[idx] = shared_buffer[tx] * 2;

            // Release the stage
            pipe.consumer_release();
        }
    }
}

// Improved version with overlapping (Wait Prior)
// This pattern issues the NEXT batch while computing the CURRENT batch.
__global__ void async_copy_overlap_kernel(const int* __restrict__ global_data, int* results, int N) {
    extern __shared__ int shared_buffer[];
    // We need double buffering in shared memory to truly overlap
    // Buffer 0: [0, blockDim.x)
    // Buffer 1: [blockDim.x, 2*blockDim.x)
    int* buffer0 = shared_buffer;
    int* buffer1 = &shared_buffer[blockDim.x];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    int tx = threadIdx.x;

    // Prologue: Issue the first batch (Batch 0)
    // Batch 0 goes into buffer0
    int idx_next = tx;
    if (idx_next < N) {
        pipe.producer_acquire();
        cuda::memcpy_async(&buffer0[tx], &global_data[idx_next], sizeof(int), pipe);
        pipe.producer_commit();
    }

    // Main Loop
    // i is the start of the NEXT batch to load (Batch 1, 2, ...)
    for (int i = blockDim.x; i < N; i += blockDim.x) {
        int idx_load = i + tx;            // Index to load next
        int idx_compute = (i - blockDim.x) + tx; // Index to compute (previous)

        // Decide buffers:
        // Batch 0 (idx_compute < blockDim.x) was in buffer0.
        // Batch 1 (idx_load) will go to buffer1.
        // Batch 2 will go to buffer0.
        // Parity 0 -> Load to buffer0 (Compute from buffer1)
        // Parity 1 -> Load to buffer1 (Compute from buffer0)
        // Note: For i=blockDim (Batch 1), parity is 1. We load to buffer1, compute from buffer0.
        int parity = (i / blockDim.x) % 2;
        int* load_ptr = (parity == 0) ? buffer0 : buffer1;
        int* compute_ptr = (parity == 0) ? buffer1 : buffer0;

        if (idx_load < N) {
            pipe.producer_acquire();
            cuda::memcpy_async(&load_ptr[tx], &global_data[idx_load], sizeof(int), pipe);
            pipe.producer_commit();
        }

        // Compute the PREVIOUS batch.
        // We need to wait until the previous stage is ready.
        // Since we just committed a new stage, we have 2 active stages (current load + previous load).
        // wait_prior<1> waits until only 1 stage is pending (the new load).
        // This effectively ensures the older stage is complete.
        if (idx_compute < N) {
            pipe.consumer_wait(cuda::pipeline_consumer_wait_prior<1>());

            results[idx_compute] = compute_ptr[tx] * 2;

            pipe.consumer_release();
        }
    }

    // Epilogue: Process the final batch
    // The loop finishes when we can no longer load a full batch (or partial batch if i >= N).
    // But we still have the last committed batch pending in the pipe.

    // Calculate which buffer holds the last batch
    // If the loop ended at i=N, the last batch loaded was N-blockDim.x.
    // Its index was (N-1)/blockDim * blockDim.
    int last_batch_start = (N - 1) / blockDim.x * blockDim.x;
    int idx_final = last_batch_start + tx;

    // Parity of the last batch loaded
    int last_parity = (last_batch_start / blockDim.x) % 2;
    int* final_compute_ptr = (last_parity == 0) ? buffer0 : buffer1;

    if (idx_final < N) {
        // Drain everything (should be just 1 stage left)
        pipe.consumer_wait();

        results[idx_final] = final_compute_ptr[tx] * 2;

        pipe.consumer_release();
    }
}

class AsyncCopyDemo {
public:
    static void run_demo() {
        printf("=== Asynchronous Data Copy (cp.async) Demo ===\n");

        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);

        if (props.major < 8) {
            printf("Skipping demo: Requires Compute Capability 8.0+ (Ampere). Detected: %d.%d\n",
                   props.major, props.minor);
            return;
        }

        const int N = 1024;
        std::vector<int> h_data(N, 1);
        std::vector<int> h_results(N, 0);

        int *d_data, *d_results;
        cudaMalloc(&d_data, N * sizeof(int));
        cudaMalloc(&d_results, N * sizeof(int));

        cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        // Shared memory size:
        // Simple kernel needs 1x block size
        // Overlap kernel needs 2x block size (double buffering)
        size_t shmem_size = 2 * 256 * sizeof(int);

        // Run Simple Kernel
        printf("Running Simple Async Copy...\n");
        async_copy_kernel<<<1, 256, shmem_size>>>(d_data, d_results, N);
        cudaDeviceSynchronize();

        // Verify Simple Kernel
        cudaMemcpy(h_results.data(), d_results, N * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Simple Kernel Result: %d (Expected: 2)\n", h_results[0]);

        // Reset results
        cudaMemset(d_results, 0, N * sizeof(int));

        // Run Overlap Kernel
        printf("Running Overlap Async Copy...\n");
        async_copy_overlap_kernel<<<1, 256, shmem_size>>>(d_data, d_results, N);
        cudaDeviceSynchronize();

        // Verify Overlap Kernel
        cudaMemcpy(h_results.data(), d_results, N * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Overlap Kernel Result: %d (Expected: 2)\n", h_results[0]);
        printf("Overlap Kernel Last Result: %d (Expected: 2)\n", h_results[N-1]);

        cudaFree(d_data);
        cudaFree(d_results);
    }
};

#endif // ASYNC_COPY_DEMO_CUH
