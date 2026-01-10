#ifndef ATOMIC_OPERATIONS_CUH
#define ATOMIC_OPERATIONS_CUH

#include <cuda_runtime.h>
#include <cstdio>

// Atomic Add Demo
__global__ void atomic_add_kernel(int* counter) {
    atomicAdd(counter, 1);
}

// Atomic CAS Demo
__global__ void atomic_cas_kernel(int* lock, int* data) {
    if (threadIdx.x == 0) {
        // Try to acquire lock (0 -> 1)
        while (atomicCAS(lock, 0, 1) != 0) {
            // Spin wait
        }

        // Critical section
        *data += 1;
        printf("Critical section executed by block %d\n", blockIdx.x);

        // Release lock (1 -> 0)
        atomicExch(lock, 0);
    }
}

class AtomicOperationsDemo {
public:
    static void run_demo() {
        printf("=== Atomic Operations Demo ===\n");

        int* d_counter;
        cudaMalloc(&d_counter, sizeof(int));
        cudaMemset(d_counter, 0, sizeof(int));

        atomic_add_kernel<<<10, 128>>>(d_counter);
        cudaDeviceSynchronize();

        int h_counter;
        cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Counter value: %d (Expected: 1280)\n", h_counter);

        cudaFree(d_counter);
    }
};

#endif // ATOMIC_OPERATIONS_CUH
