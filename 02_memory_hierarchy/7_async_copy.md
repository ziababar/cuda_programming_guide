# Asynchronous Memory Copy (`cp.async`)

Introduced in CUDA 11 for Ampere (Compute Capability 8.0+) and later architectures, Asynchronous Memory Copy (`cp.async`) is a hardware-accelerated mechanism to move data from Global Memory to Shared Memory without using the register file as an intermediate buffer.

**[Back to Memory Hierarchy](1_cuda_memory_hierarchy.md)** | **Previous: [Memory Debugging](6_memory_debugging.md)**

---

## **Why `cp.async`?**

Traditional memory copies (Global → Register → Shared) consume:
1.  **Register File Bandwidth**: Data must pass through registers.
2.  **Register Count**: Registers are occupied while waiting for global memory.
3.  **Instruction Latency**: Load and Store instructions are issued sequentially.

**`cp.async` bypasses the register file**, moving data directly from L2 Cache/DRAM to Shared Memory. This frees up registers for computation and allows for better latency hiding via asynchronous pipelines.

---

## **Key Concepts**

1.  **Direct Transfer**: Global Memory $\rightarrow$ Shared Memory (bypassing registers).
2.  **Asynchronous**: The copy command is issued, and the thread can continue execution.
3.  **Pipeline**: Organize copies into stages (commit, wait) to overlap data movement with computation.

---

## **C++ API: `cuda::memcpy_async`**

The `cuda::memcpy_async` API (part of `libcu++`) provides a C++ abstraction over the PTX `cp.async` instructions.

```cpp
#include <cuda/pipeline>

__global__ void async_copy_kernel(int* global_data, int* shared_buffer) {
    // Create a pipeline object
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Issue asynchronous copy
    pipe.producer_acquire();
    cuda::memcpy_async(&shared_buffer[threadIdx.x],
                       &global_data[threadIdx.x],
                       sizeof(int),
                       pipe);
    pipe.producer_commit();

    // Wait for the copy to complete
    pipe.consumer_wait();
    pipe.consumer_release();

    // Data is now ready in shared_buffer
}
```

---

## **Pipeline Pattern: Multi-Stage Pipelining**

The most powerful use case is overlapping computation with memory transfers for the *next* iteration (software pipelining).

```mermaid
graph LR
    subgraph "Iteration N"
        Compute_N[Compute Tile N]
    end

    subgraph "Iteration N+1"
        Load_N1[Load Tile N+1 (Async)]
    end

    Compute_N -- Overlaps with --> Load_N1
```

### **Code Example: Double Buffering**

See `src/02_memory_hierarchy/AsyncCopyDemo.cuh` for a complete implementation.

```cpp
// Pseudo-code for a 2-stage pipeline
for (int i = 0; i < N; i += TILE_SIZE) {
    // Stage 1: Issue Copy for NEXT tile
    pipe.producer_acquire();
    cuda::memcpy_async(shared_next, global_next, size, pipe);
    pipe.producer_commit();

    // Stage 2: Compute CURRENT tile (already loaded)
    compute(shared_current);

    // Stage 3: Wait for NEXT tile to arrive
    pipe.consumer_wait();

    // Swap buffers
    swap(shared_current, shared_next);
    pipe.consumer_release();
}
```

---

## **Requirements & Constraints**

1.  **Compute Capability**: Requires SM 8.0 (Ampere) or higher.
2.  **Alignment**: 128-byte alignment (e.g., `float4`) is optimal for maximum bandwidth.
3.  **Compilation**: Must compile with `-arch=sm_80` or higher.

---

## **Performance Impact**

Using `cp.async` can significantly improve performance for memory-bound kernels (like GEMM, Convolution) by:
*   Reducing register pressure.
*   Maximizing global memory bandwidth.
*   Perfectly overlapping compute and memory latency.

---

## **Related Guides**

*   **[Global Memory](2_global_memory.md)** - Traditional access patterns.
*   **[Shared Memory](3_shared_memory.md)** - Understanding shared memory banks.
