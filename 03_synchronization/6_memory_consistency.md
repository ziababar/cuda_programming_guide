#  Memory Consistency

Understanding memory consistency models is crucial for correct synchronization across different memory types and scopes.

**Previous: [Atomic Operations](5_atomic_operations.md)** | **Next: [Debugging Synchronization](7_synchronization_debugging.md)**

---

##  **Memory Consistency**

###  **Memory Ordering**

#### **Memory Fence Operations:**
```cpp
// Memory fence usage for consistency
__global__ void memory_consistency_demo(int* flags, int* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Producer thread
        if (tid % 2 == 0) {
            // Write data first
            data[tid] = tid * 100;

            // Memory fence ensures data write completes before flag write
            __threadfence();  // Global memory fence

            // Signal data is ready
            flags[tid] = 1;
        }
        // Consumer thread
        else {
            int partner = tid - 1;

            // Busy wait for flag
            while (atomicAdd(&flags[partner], 0) == 0) {
                // Spin wait
            }

            // Memory fence ensures flag read completes before data read
            __threadfence();

            // Now safe to read data
            int value = data[partner];
            printf("Thread %d read value %d\n", tid, value);
        }
    }
}

// Different fence scopes
__global__ void memory_fence_scopes(int* global_data, int N) {
    __shared__ int shared_data[256];
    int tid = threadIdx.x;

    // Thread fence (affects all memory spaces)
    shared_data[tid] = tid;
    global_data[tid] = tid * 2;
    __threadfence();  // Ensures both writes are visible

    // Block fence (affects memory visible to thread block)
    __threadfence_block();  // Ensures shared memory consistency

    // System fence (affects system-wide memory)
    __threadfence_system();  // Ensures visibility to CPU and other GPUs
}
```

#### **Relaxed Memory Model Programming:**
```cpp
// Handle relaxed memory ordering correctly
__global__ void relaxed_memory_programming(volatile int* sync_flags,
                                          int* shared_buffer, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_id = blockIdx.x;

    if (tid < N) {
        // Each block acts as producer or consumer
        if (block_id % 2 == 0) {
            // Producer block
            shared_buffer[tid] = tid + 1000;

            // Ensure write completes
            __threadfence();

            // Signal completion using volatile (prevents compiler optimization)
            sync_flags[block_id] = 1;
        }
        else {
            // Consumer block
            int producer_id = block_id - 1;

            // Wait for producer signal
            while (sync_flags[producer_id] == 0) {
                // Volatile access prevents optimization
            }

            // Read data produced by other block
            int consumed_value = shared_buffer[tid - blockDim.x];

            printf("Block %d thread %d consumed: %d\n",
                   block_id, threadIdx.x, consumed_value);
        }
    }
}
```
