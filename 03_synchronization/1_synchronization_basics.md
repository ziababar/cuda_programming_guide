#  Synchronization Fundamentals

Synchronization is the cornerstone of parallel programming in CUDA. Understanding how threads coordinate, share data safely, and maintain consistency is essential for writing correct and efficient GPU applications.

**[Back to Execution Model](../01_execution_model/1_cuda_execution_model.md)** | **Previous: [Streaming Multiprocessors Guide](../01_execution_model/4_streaming_multiprocessors_deep.md)** | **Next: [Block-Level Synchronization](2_block_synchronization.md)**

---

##  **Synchronization Fundamentals**

CUDA provides multiple levels of synchronization, each with different scopes, performance characteristics, and use cases.

###  **Synchronization Hierarchy**

| Level | Scope | Mechanism | Performance | Use Cases |
|-------|-------|-----------|-------------|-----------|
| **Thread** | Individual thread | None required | N/A | Independent operations |
| **Warp** | 32 threads | Implicit SIMT + `__syncwarp()` | Very fast | Warp-level algorithms |
| **Block** | Thread block | `__syncthreads()` | Fast | Shared memory cooperation |
| **Grid** | Entire kernel | Kernel boundaries | Slow | Multi-stage algorithms |
| **Device** | Multiple kernels | `cudaDeviceSynchronize()` | Very slow | Host-device coordination |

###  **Synchronization Models**

#### **Barrier Synchronization:**
```cpp
// Basic barrier synchronization patterns
__global__ void barrier_synchronization_demo() {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;

    // Phase 1: All threads write to shared memory
    shared_data[tid] = tid * 2.0f;

    // Barrier: Ensure all writes complete before reads
    __syncthreads();

    // Phase 2: All threads can safely read any shared memory location
    float neighbor_value = shared_data[(tid + 1) % blockDim.x];
    shared_data[tid] = shared_data[tid] + neighbor_value;

    // Another barrier before final phase
    __syncthreads();

    // Phase 3: Final processing with consistent shared memory state
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[i];
        }
        printf("Block sum: %.2f\n", sum);
    }
}
```

#### **Parallel Reduction (Safe Block Sync Pattern):**
```cpp
// Correct pattern: All threads participate in the barrier
__global__ void reduction_demo(float* input, float* output) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load data
    sdata[tid] = input[idx];
    __syncthreads(); // Barrier 1: Wait for load

    // Reduction in shared memory
    // Note: All threads execute the loop, even if they don't do work in later iterations.
    // However, __syncthreads() MUST be reached by all active threads in the block.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        // Essential: All threads reach this barrier in every iteration
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

> **Warning**: Avoid placing `__syncthreads()` inside divergent branches (e.g., `if (tid < 16) { ... __syncthreads(); } else { ... }`). This can cause deadlocks because the barrier waits for *all* threads in the block to reach it.
