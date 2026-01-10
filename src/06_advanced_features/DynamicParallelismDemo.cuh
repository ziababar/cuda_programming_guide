#ifndef DYNAMIC_PARALLELISM_DEMO_CUH
#define DYNAMIC_PARALLELISM_DEMO_CUH

#include <cuda_runtime.h>
#include <cstdio>

// Child kernel
__global__ void child_kernel(int depth) {
    printf("  Child kernel at depth %d (Block %d, Thread %d)\n", depth, blockIdx.x, threadIdx.x);
}

// Parent kernel
__global__ void parent_kernel(int max_depth) {
    if (threadIdx.x == 0) {
        printf("Parent kernel launching child (Depth 0)\n");

        // Launch child kernel
        // Note: Dynamic parallelism requires relocatable device code (-rdc=true)
        child_kernel<<<2, 2>>>(1);

        // Wait for child to complete
        cudaDeviceSynchronize();

        printf("Parent kernel resumes after child completes\n");
    }
}

class DynamicParallelismDemo {
public:
    static void run_demo() {
        printf("=== Dynamic Parallelism Demo ===\n");
        printf("Note: Compilation requires -rdc=true\n");

        parent_kernel<<<1, 32>>>(1);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
             printf("CUDA Error (Check if -rdc=true is enabled): %s\n", cudaGetErrorString(err));
        }
    }
};

#endif // DYNAMIC_PARALLELISM_DEMO_CUH
