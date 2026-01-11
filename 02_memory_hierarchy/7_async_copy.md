# Asynchronous Memory Copy (`cp.async`)

Introduced in the Ampere architecture (Compute Capability 8.0), Asynchronous Memory Copy (`cp.async`) is a powerful feature that allows data to be copied from global memory directly into shared memory without blocking the execution of the thread.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Next: [Synchronization](../03_synchronization/1_synchronization_basics.md)**

---

## **The Memory Latency Problem**

In traditional CUDA kernels, loading data from global memory is a blocking operation for the thread (or at least consumes register resources and instruction slots):

```cpp
// Traditional Load
float val = global_ptr[idx]; // Loads into register
__syncthreads();             // Wait (if other threads also load)
shared_ptr[tid] = val;       // Store to shared memory
```

This approach has two downsides:
1.  **Register Pressure**: Data must pass through the Register File (RF) before going to Shared Memory (SMEM).
2.  **Stalls**: The warp may stall waiting for the load to complete if not enough independent arithmetic instructions are available to hide the latency.

## **The `cp.async` Solution**

`cp.async` (exposed via `cuda::memcpy_async` in C++) creates a direct path from L2 Cache/Global Memory to Shared Memory, bypassing the Register File.

### **Key Benefits**
*   **Reduced Register Pressure**: Data doesn't consume thread registers.
*   **Latency Hiding**: The copy happens in the background while the thread continues executing other instructions.
*   **Bandwidth Efficiency**: Optimizes the data path within the SM.

---

## **Using `cuda::pipeline`**

The modern C++ interface for `cp.async` is `cuda::pipeline`. It provides a structured way to manage asynchronous copies.

### **Basic Pipeline Workflow**

1.  **Acquire**: Reserve space in the pipeline for a new batch of copies.
2.  **Copy**: Issue asynchronous copy commands.
3.  **Commit**: Signal that all copies for this batch have been issued.
4.  **Wait**: Block until a specific batch is ready.

### **Code Example**

See [`AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh) for a complete compilable example.

```cpp
#include <cuda_pipeline.h>

__global__ void pipeline_example(float* global_in, float* global_out) {
    // 1. Create a pipeline object
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Shared memory buffer
    __shared__ float smem[128];

    // 2. Issue Async Copy
    pipe.producer_acquire();
    cuda::memcpy_async(&smem[threadIdx.x], &global_in[threadIdx.x], sizeof(float), pipe);
    pipe.producer_commit();

    // ... Do other independent work here ...

    // 3. Wait for the copy to finish
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    pipe.consumer_release();

    // 4. Use data
    global_out[threadIdx.x] = smem[threadIdx.x] * 2.0f;
}
```

---

## **Multi-Stage Pipelining**

To fully saturate memory bandwidth, you often use **Multi-Stage Pipelining** (e.g., Double Buffering). While one buffer is being consumed (computed on), the next buffer is being pre-fetched.

### **Visualizing the Pipeline**

| Time Step | Buffer 0 | Buffer 1 | Action |
| :--- | :--- | :--- | :--- |
| **T0** | **Load** (Async) | | Issue Load for Batch 0 |
| **T1** | *Wait* | **Load** (Async) | Wait for Batch 0, Issue Load for Batch 1 |
| **T2** | **Compute** | *Wait* | Compute on Batch 0, Wait for Batch 1 |
| **T3** | | **Compute** | Compute on Batch 1 |

### **Implementation Pattern**

```cpp
// Prologue: Start first batch
pipe.producer_acquire();
cuda::memcpy_async(buffer[0], src, size, pipe);
pipe.producer_commit();

// Loop
for (int i = 0; i < N; ++i) {
    // Wait for current batch (i % 2)
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    pipe.consumer_release();

    // Issue PREFETCH for next batch ((i + 1) % 2)
    pipe.producer_acquire();
    cuda::memcpy_async(buffer[(i+1)%2], next_src, size, pipe);
    pipe.producer_commit();

    // Compute on current batch (i % 2)
    process(buffer[i % 2]);
}
```

---

## **Hardware Requirements**

*   **Compute Capability**: 8.0 (Ampere) or higher.
*   **CUDA Toolkit**: 11.0 or higher.

## **Performance Tips**

1.  **Alignment**: Ensure source and destination pointers are aligned (ideally 16 bytes) for maximum efficiency (`cp.async.cg`).
2.  **Batching**: Group multiple `memcpy_async` calls into a single commit to reduce overhead.
3.  **Warp-Level vs Thread-Level**: You can use `cuda::pipeline<cuda::thread_scope_block>` for cooperative copying where all threads in a block participate in the pipeline management.

---

**Related Guides**
*   [Global Memory](2_global_memory.md)
*   [Shared Memory](3_shared_memory.md)
