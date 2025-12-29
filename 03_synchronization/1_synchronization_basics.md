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

### **Memory Ordering & Fences**

Synchronization is not just about timing; it's about **memory visibility**.

- **`__syncthreads()`**: Barrier for all threads in a block + memory fence (shared/global).
- **`__threadfence_block()`**: Ensures memory writes are visible to all threads in the *block*.
- **`__threadfence()`**: Ensures memory writes are visible to all threads in the *device*.
- **`__threadfence_system()`**: Ensures memory writes are visible to *entire system* (Host + Device).

> **Crucial**: Fences do *not* pause threads (like `__syncthreads()` does). They only enforce memory ordering.

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

#### **Modern Synchronization (C++20 / CUDA 11+)**
CUDA 11 introduced `cuda::barrier` and `cuda::pipeline` for more flexible synchronization, similar to C++20 standard library features.

```cpp
#include <cuda/barrier>
#include <cuda/pipeline>

__global__ void modern_sync_demo(float* data, int N) {
    // Basic block barrier using libcu++
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) {
        bar.init(blockDim.x);
    }
    __syncthreads();

    // Do work...
    // Arrive and wait
    bar.arrive_and_wait();
}
```

#### **Producer-Consumer Pattern:**
```cpp
// Producer-consumer synchronization with shared memory
__global__ void producer_consumer_demo() {
    __shared__ float buffer[128];
    __shared__ int write_pos;
    __shared__ int read_pos;
    __shared__ int data_count;

    int tid = threadIdx.x;

    // Initialize shared variables
    if (tid == 0) {
        write_pos = 0;
        read_pos = 0;
        data_count = 0;
    }
    __syncthreads();

    // Producer threads (first half)
    if (tid < blockDim.x / 2) {
        for (int i = 0; i < 4; i++) {  // Each producer creates 4 items
            float data = tid * 4 + i;

            // Wait for buffer space
            while (data_count >= 128) {
                __syncthreads();
            }

            // Produce data
            int pos = atomicAdd(&write_pos, 1) % 128;
            buffer[pos] = data;
            atomicAdd(&data_count, 1);

            __syncthreads();
        }
    }
    // Consumer threads (second half)
    else {
        for (int i = 0; i < 2; i++) {  // Each consumer processes 2 items
            // Wait for data
            while (data_count <= 0) {
                __syncthreads();
            }

            // Consume data
            int pos = atomicAdd(&read_pos, 1) % 128;
            float data = buffer[pos];
            atomicSub(&data_count, 1);

            // Process data
            printf("Consumer %d processed: %.1f\n", tid, data);

            __syncthreads();
        }
    }
}
```
