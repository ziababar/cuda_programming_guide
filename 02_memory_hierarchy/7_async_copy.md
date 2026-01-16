# Asynchronous Memory Copy (`cp.async`)

Asynchronous Memory Copy, introduced in NVIDIA Ampere architecture (Compute Capability 8.0), is a powerful feature that allows data to be moved from Global Memory directly to Shared Memory without passing through the register file.

**[Back to Overview](1_cuda_memory_hierarchy.md)** | **Previous: [Memory Debugging](6_memory_debugging.md)**

---

## **The Problem: Latency & Register Pressure**

In pre-Ampere architectures, loading data from Global Memory to Shared Memory required an intermediate stop in the thread's registers:

1.  **Load**: Global Memory $\to$ Register (High Latency)
2.  **Store**: Register $\to$ Shared Memory
3.  **Sync**: `__syncthreads()` to ensure data visibility.

This approach has two downsides:
-   **Register Usage**: Consumes registers just to move data.
-   **Blocking**: Threads often stall waiting for the global load to complete before they can store to shared memory (unless complex pre-fetching is used).

## **The Solution: `cp.async`**

With `cp.async`, the memory transfer is offloaded to a dedicated hardware unit (Memory Copy Engine). The data bypasses the execution cores and registers:

`Global Memory` $\xrightarrow{\text{cp.async}}$ `L2 Cache` $\xrightarrow{\text{cp.async}}$ `Shared Memory`

**Benefits:**
-   **Zero Register Usage**: Does not consume general-purpose registers for the data path.
-   **Compute-Copy Overlap**: Threads can continue computing while the copy engine handles the transfer.
-   **Bandwidth Efficiency**: Reduces traffic on the register file ports.

---

## **The `cuda::pipeline` API**

CUDA 11 introduced the C++ `cuda::pipeline` interface to manage these asynchronous operations safely. It uses a **multi-stage pipeline** model.

### **Pipeline Stages**
A pipeline consists of "stages". Each stage represents a batch of memory operations.
1.  **Acquire**: Open a new stage (`producer_acquire`).
2.  **Issue**: Enqueue async copy commands (`memcpy_async`).
3.  **Commit**: Close the stage (`producer_commit`).
4.  **Wait**: Wait for a specific stage to complete (`consumer_wait`).
5.  **Release**: improved reuse of resources (`consumer_release`).

---

## **Implementation Example**

Here is a double-buffered pipeline pattern. While the GPU computes on **Buffer A**, it asynchronously loads the next batch of data into **Buffer B**.

**Full Source Code:** [`src/02_memory_hierarchy/AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh)

```cpp
#include <cuda/pipeline>

template <int BLOCK_SIZE>
__global__ void async_copy_kernel(float* d_out, const float* d_in, int N) {
    // 1. Allocate Double Buffer in Shared Memory
    __shared__ float buffer[2][BLOCK_SIZE];

    // 2. Create Pipeline
    cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline();

    // ... setup indices ...
    int view = 0; // Current buffer index

    // PROLOGUE: Pre-load the first batch
    pipe.producer_acquire();
    cuda::memcpy_async(&buffer[view][threadIdx.x], &d_in[...], sizeof(float), pipe);
    pipe.producer_commit();

    // MAIN LOOP
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        // A. Issue load for NEXT batch (into view ^ 1)
        pipe.producer_acquire();
        cuda::memcpy_async(&buffer[view ^ 1][threadIdx.x], &d_in[...], sizeof(float), pipe);
        pipe.producer_commit();

        // B. Wait for CURRENT batch (view) to be ready
        // wait_prior<1> means "wait until only 1 stage (the one we just issued) is pending"
        // This ensures the PREVIOUS stage (the one we want to compute on) is complete.
        pipe.consumer_wait();

        // Sync block to ensure all threads see the shared memory data
        __syncthreads();

        // C. Compute on Current Data (buffer[view])
        d_out[...] = buffer[view][threadIdx.x] * 2.0f;

        // D. Release stage
        pipe.consumer_release();

        // Swap buffers
        view ^= 1;
    }
}
```

---

## **Performance Considerations**

1.  **Ampere Requirement**: This feature works only on Compute Capability 8.0 and higher (A100, RTX 30/40 series).
2.  **Tiling**: Works best when `sizeof(T)` is 4, 8, or 16 bytes. 16-byte transfers (e.g., `float4`) are the most efficient.
3.  **Occupancy**: Because `cp.async` reduces register pressure, it can sometimes allow for higher occupancy (more active warps) compared to traditional loads.
4.  **Shared Memory Bandwidth**: While loading is efficient, ensure you avoid bank conflicts when *reading* the data in the compute phase (see [Shared Memory Guide](3_shared_memory.md)).

---

## **Related Guides**

*   **[Shared Memory Guide](3_shared_memory.md)**: Deep dive into shared memory optimization.
*   **[Global Memory Guide](2_global_memory.md)**: Understanding coalescing which still applies to the source address in `cp.async`.
