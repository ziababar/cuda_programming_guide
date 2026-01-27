# Asynchronous Memory Copy

Asynchronous Memory Copy (`cp.async`) is a powerful feature introduced in the NVIDIA Ampere architecture (Compute Capability 8.0). It allows data to be copied from Global Memory directly to Shared Memory without passing through the register file, and importantly, without blocking the execution thread.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **[Back to Memory Index](1_cuda_memory_hierarchy.md)**

---

## **Why Asynchronous Copy?**

Traditional memory loads (`LDG`) block the execution thread (or warp) until the data arrives (or until the instruction window fills up). They also consume register resources to hold the data before it's written to shared memory (`STS`).

**Benefits of `cp.async`:**
1.  **Latency Hiding**: The copy happens in the background. The thread can execute other independent instructions (math, index calculation) while waiting for memory.
2.  **Reduced Register Pressure**: Data goes `Global -> Shared`, bypassing registers. This frees up registers for compute, potentially increasing occupancy.
3.  **Bandwidth Efficiency**: It maps directly to hardware-accelerated asynchronous copy engines in the SM.

---

## **The C++ Pipeline Interface**

CUDA 11 introduced the `<cuda/pipeline>` header (part of libcu++) to provide a standard C++ interface for asynchronous operations.

### **The Pipeline Pattern**

The standard workflow involves a multi-stage pipeline:

1.  **Acquire**: Reserve resources (a "stage") in the pipeline.
2.  **Copy**: Issue asynchronous copy commands (`cuda::memcpy_async`).
3.  **Commit**: Mark the end of the batch of copy commands.
4.  **Wait**: Block the thread until a specific stage is complete.
5.  **Release**: Release the stage resources.

### **Code Example**

See [`src/02_memory_hierarchy/AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh) for a complete compilation-ready example.

```cpp
#include <cuda/pipeline>

__global__ void async_copy_kernel(float* d_out, const float* d_in, int N) {
    extern __shared__ float shared_mem[];
    float* tile = shared_mem;

    // Create a pipeline object for this thread
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        // 1. ACQUIRE: Reserve space in the pipeline
        pipe.producer_acquire();

        // 2. COPY: Issue asynchronous copy from Global to Shared Memory
        // This bypasses the register file
        cuda::memcpy_async(&tile[tid], &d_in[idx], sizeof(float), pipe);

        // 3. COMMIT: Commit the copy command
        pipe.producer_commit();

        // --- Overlap Opportunity ---
        // Perform independent math here while memory is fetching!
        // ---------------------------

        // 4. WAIT: Wait for the oldest stage to finish
        pipe.consumer_wait();

        // 5. CONSUME: Use the data
        float val = tile[tid];
        d_out[idx] = val * 2.0f;

        // 6. RELEASE: Release the stage
        pipe.consumer_release();
    }
}
```

---

## **Multi-Stage Pipelines**

Real-world performance gains come from **multi-stage pipelines** (e.g., double buffering or circular buffers). While one stage is loading, another stage is computing.

```cpp
// Pseudo-code for a 2-stage pipeline
for (int i = 0; i < num_tiles; ++i) {
    // 1. Issue Load for Next Tile
    pipe.producer_acquire();
    cuda::memcpy_async(next_tile_ptr, global_ptr, size, pipe);
    pipe.producer_commit();

    // 2. Wait for Current Tile (issued in previous iteration)
    pipe.consumer_wait();

    // 3. Compute on Current Tile
    compute(current_tile_ptr);

    // 4. Release Current Tile
    pipe.consumer_release();

    // Swap pointers
    swap(current_tile_ptr, next_tile_ptr);
}
```

---

## **Key Takeaways**

1.  **Ampere+ Only**: Requires Compute Capability 8.0 or higher (`-arch=sm_80`).
2.  **Bypass Registers**: Saves register usage by routing data `Global -> Shared`.
3.  **Standard API**: Use `cuda::pipeline` and `cuda::memcpy_async` for a portable, modern C++ implementation.
4.  **Overlap**: The primary goal is to overlap "Compute" with "Memory Transfer".

---

## **Related Guides**

- **[Shared Memory Guide](3_shared_memory.md)**: Understanding shared memory banks and layout.
- **[Streams & Concurrency](../04_streams_concurrency/1_stream_fundamentals.md)**: Managing concurrency at the kernel level.
