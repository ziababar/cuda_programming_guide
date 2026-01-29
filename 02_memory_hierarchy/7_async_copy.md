# Asynchronous Memory Copy (`cp.async`)

Asynchronous Memory Copy, introduced in the NVIDIA Ampere Architecture (Compute Capability 8.0), is a powerful feature that allows threads to issue copy commands from Global Memory to Shared Memory without blocking the execution pipeline. This enables significant performance improvements by overlapping memory transfer with computation.

**[Back to Overview](1_cuda_memory_hierarchy.md)** | **Previous: [Memory Debugging](6_memory_debugging.md)**

---

## **Table of Contents**

1. [Concept: Direct Global-to-Shared Copy](#-concept-direct-global-to-shared-copy)
2. [The `cuda::pipeline` Interface](#-the-cudapipeline-interface)
3. [Implementation Guide](#-implementation-guide)
4. [Double Buffering Pattern](#-double-buffering-pattern)
5. [Performance Considerations](#-performance-considerations)

---

## **Concept: Direct Global-to-Shared Copy**

In previous architectures (Volta and older), loading data from Global Memory to Shared Memory required the data to pass through the register file:

1.  **Load:** Global Memory → Register
2.  **Store:** Register → Shared Memory

This consumed register bandwidth and required the thread to wait for the load to complete before storing.

**With `cp.async` (Ampere+):**
The copy is issued by the SM (Streaming Multiprocessor) to the memory controller. The data goes **directly** from Global Memory to Shared Memory, bypassing the register file entirely.

*   **Reduces Register Pressure:** Frees up registers for computation.
*   **Hides Latency:** Threads can perform other work (e.g., math) while data is moving.
*   **High Bandwidth:** Direct data path utilizes full memory bandwidth.

---

## **The `cuda::pipeline` Interface**

Modern CUDA C++ (CUDA 11.0+) provides the `cuda::pipeline` synchronization primitive (in `<cuda/pipeline>`) to manage these asynchronous copies.

### **Key Operations**

| Operation | Description |
|-----------|-------------|
| `cuda::make_pipeline()` | Creates a pipeline object. |
| `producer_acquire()` | Acquires a stage in the pipeline for issuing copy commands. |
| `cuda::memcpy_async()` | Issues the asynchronous copy (Global → Shared). |
| `producer_commit()` | Marks the current stage as "committed" (transfer started). |
| `consumer_wait()` | Blocks until the specified stage is complete. |
| `consumer_release()` | Releases the stage after the consumer (computation) is done. |

---

## **Implementation Guide**

Below is a complete example of using `cp.async` with double buffering to hide memory latency.

> **Source Code:** The full compilable example is available in [`../src/02_memory_hierarchy/AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh).

### **Basic Pipeline Structure**

```cpp
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void async_copy_kernel(float* global_in, float* global_out, int N) {
    __shared__ float shared_buffer[BLOCK_SIZE];

    auto block = cg::this_thread_block();
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    // 1. Acquire stage
    pipeline.producer_acquire();

    // 2. Issue Async Copy
    // copy size bytes from global_in to shared_buffer
    cuda::memcpy_async(shared_buffer, global_in, sizeof(float) * BLOCK_SIZE, pipeline);

    // 3. Commit stage
    pipeline.producer_commit();

    // 4. Wait for copy to finish
    pipeline.consumer_wait();

    // 5. Compute (safe to read shared_buffer)
    // ... computation ...

    // 6. Release stage
    pipeline.consumer_release();
}
```

---

## **Double Buffering Pattern**

To truly maximize performance, you should use **Multi-Stage Pipelining** (Double or Triple Buffering). While the GPU is computing on Batch `N`, it is simultaneously loading Batch `N+1`.

### **Code Example**

```cpp
// Double Buffering: 2 buffers
__shared__ float buffer[2][BLOCK_SIZE];

// Prologue: Start loading Batch 0
pipeline.producer_acquire();
cuda::memcpy_async(buffer[0], &in[0], size, pipeline);
pipeline.producer_commit();

// Main Loop
for (int i = 0; i < num_batches; ++i) {
    int curr = i % 2;
    int next = (i + 1) % 2;

    // Prefetch Batch i+1 (if exists)
    if (i < num_batches - 1) {
        pipeline.producer_acquire();
        cuda::memcpy_async(buffer[next], &in[(i + 1) * BLOCK_SIZE], size, pipeline);
        pipeline.producer_commit();
    }

    // Wait for Batch i to complete
    pipeline.consumer_wait();

    // Compute on Batch i
    process(buffer[curr]);

    // Release Batch i
    pipeline.consumer_release();
}
```

---

## **Performance Considerations**

1.  **Compute Capability:** This feature requires **SM 8.0 (Ampere)** or higher (e.g., A100, RTX 30/40 series). On older hardware, `cuda::memcpy_async` falls back to synchronous copies (no performance gain).
2.  **Pinned Memory:** For host-to-device transfers to be truly asynchronous at the system level, host memory must be pinned (`cudaHostAlloc`).
3.  **Warp Coalescing:** Like standard global loads, `cp.async` works best when the warp accesses contiguous memory.
4.  **Shared Memory Banks:** Ensure the destination in shared memory avoids bank conflicts, just like regular stores.
5.  **Compiler Flags:** Compile with `-arch=sm_80` (or higher) and C++17 support (`-std=c++17` is default in modern NVCC for this).

### **When to Use**
*   **Memory Bound Kernels:** Where hiding global memory latency is critical.
*   **GEMM / Matrix Multiplication:** Essential for feeding Tensor Cores efficiently.
*   **Stencil Codes:** Pre-loading halos or blocks.

---

**Next Steps:**
*   Explore [Tensor Cores](../06_advanced_features/1_tensor_cores.md) which heavily rely on `cp.async`.
*   Review [Synchronization](../03_synchronization/1_synchronization_basics.md) primitives.
