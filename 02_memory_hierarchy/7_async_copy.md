# Asynchronous Memory Copy

Introduced in the Ampere architecture (Compute Capability 8.0), Asynchronous Memory Copy (`cp.async`) allows threads to initiate data transfers from Global Memory to Shared Memory without blocking execution.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Next: [Synchronization](../03_synchronization/1_synchronization_basics.md)**

---

## **Overview**

Traditional memory loads (`LDG`) move data from Global Memory → Registers → Shared Memory. This consumes register file bandwidth and forces threads to wait for the load to complete (unless carefully manually pipelined).

**Asynchronous Copy (`cp.async`)** bypasses the register file, moving data directly from L2 Cache/Global Memory to Shared Memory.

### **Key Benefits**
1.  **Latency Hiding**: Threads can perform compute work while data is in flight.
2.  **Reduced Register Pressure**: Data goes straight to Shared Memory, freeing up registers.
3.  **Bandwidth Efficiency**: Optimizes the memory pipeline.

---

## **The `cuda::pipeline` API**

CUDA C++ provides the `cuda::pipeline` interface (in `<cuda/pipeline>`) to manage asynchronous copies.

### **Pipeline Stages**
A pipeline consists of multiple stages (typically used for double or triple buffering).
1.  **Acquire**: Reserve a spot in the pipeline for a new batch of copies.
2.  **Issue**: Submit `memcpy_async` commands.
3.  **Commit**: Mark the batch as "in-flight".
4.  **Wait**: Wait for a specific stage to complete before consuming the data.
5.  **Release**: Signal that the data has been consumed.

### **Basic Workflow**

```cpp
#include <cuda/pipeline>

__global__ void async_kernel(...) {
    // 1. Create Pipeline
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // 2. Loop
    for (...) {
        // ACQUIRE: Get space for next batch
        pipe.producer_acquire();

        // ISSUE: Async Copy
        cuda::memcpy_async(shared_ptr, global_ptr, size, pipe);

        // COMMIT: Seal the batch
        pipe.producer_commit();

        // WAIT: Wait for previous batch to finish
        pipe.consumer_wait();

        // COMPUTE: Process data
        compute(shared_data);

        // RELEASE: Free the stage
        pipe.consumer_release();
    }
}
```

---

## **Double Buffering Example**

A common pattern is **Double Buffering**, where the kernel computes on **Buffer A** while loading **Buffer B**.

See `src/02_memory_hierarchy/AsyncCopyDemo.cuh` for the complete implementation.

```cpp
// from src/02_memory_hierarchy/AsyncCopyDemo.cuh

// Prologue: Start loading Tile 0
pipe.producer_acquire();
cuda::memcpy_async(buffer[0], ..., pipe);
pipe.producer_commit();

// Main Loop
for (int i = 0; i < N; ++i) {
    int next_buf = 1 - curr_buf;

    // Trigger load for NEXT tile
    pipe.producer_acquire();
    cuda::memcpy_async(buffer[next_buf], ..., pipe);
    pipe.producer_commit();

    // Wait for CURRENT tile to arrive
    pipe.consumer_wait();

    // Compute on CURRENT tile
    block.sync(); // Optional depending on access pattern
    process(buffer[curr_buf]);
    block.sync();

    // Release stage
    pipe.consumer_release();

    // Swap
    curr_buf = next_buf;
}
```

## **Requirements**

-   **Hardware**: NVIDIA Ampere (SM_80) or newer.
-   **Compiler**: `nvcc` with `-arch=sm_80` flag.
-   **Headers**: `<cuda/pipeline>` and `<cooperative_groups.h>`.

## **Best Practices**

1.  **Batch Size**: Issue multiple `memcpy_async` calls between `acquire` and `commit` to maximize bandwidth.
2.  **Shared Memory**: Ensure sufficient shared memory allocation for all active pipeline stages (e.g., 2x TileSize for double buffering).
3.  **Synchronization**: Use `block.sync()` if threads consume data loaded by other threads in the block.
