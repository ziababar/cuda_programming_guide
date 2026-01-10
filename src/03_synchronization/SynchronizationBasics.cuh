#ifndef SYNCHRONIZATION_BASICS_CUH
#define SYNCHRONIZATION_BASICS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cuda/barrier>

// Demonstate __syncthreads()
__global__ void syncthreads_demo_kernel(int* data) {
    int tid = threadIdx.x;

    // Phase 1
    data[tid] += 1;
    __syncthreads(); // Barrier 1

    // Phase 2 (depends on Phase 1)
    if (tid > 0) {
        data[tid] += data[tid-1];
    }
    __syncthreads(); // Barrier 2

    if (tid == 0) {
        printf("Block 0 synchronization complete.\n");
    }
}

// Demonstrate cuda::barrier (Modern CUDA)
// Requires Compute Capability 8.0+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__global__ void aw_barrier_demo_kernel(int* data, int N) {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    // Work phase
    for (int i = 0; i < N; ++i) {
        // Do work
        data[threadIdx.x]++;

        // Wait for all threads to complete this iteration
        // Arrive and wait
        bar.arrive_and_wait();
    }
}
#endif

class SynchronizationBasics {
public:
    static void run_demo() {
        printf("=== Synchronization Basics Demo ===\n");
        int* d_data;
        cudaMalloc(&d_data, 256 * sizeof(int));
        cudaMemset(d_data, 0, 256 * sizeof(int));

        syncthreads_demo_kernel<<<1, 256>>>(d_data);
        cudaDeviceSynchronize();

        cudaFree(d_data);
    }
};

#endif // SYNCHRONIZATION_BASICS_CUH
