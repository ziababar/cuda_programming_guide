# Asynchronous Memory Copy (`cp.async`)

Introduced in the Ampere architecture (Compute Capability 8.0), Asynchronous Memory Copy (`cp.async` instruction) allows data to be moved from Global Memory directly to Shared Memory without passing through the Register File. This significantly reduces register pressure and allows for better overlap of compute and data transfer.

**[Back to Memory Hierarchy Index](1_cuda_memory_hierarchy.md)**

---

## 1. Introduction

In previous architectures (Volta/Turing), loading data from Global to Shared Memory involved:
1.  **Load**: Global Memory -> Register
2.  **Store**: Register -> Shared Memory

This occupied registers and required the thread to wait for the load to complete (or use instruction-level parallelism).

With **`cp.async`**:
1.  **Copy**: Global Memory -> Shared Memory (Bypassing Register File)

This is non-blocking, allowing the thread to issue other instructions (including other copies) before waiting.

---

## 2. Using `cuda::pipeline`

The easiest way to use `cp.async` in CUDA C++ is via the `cuda::pipeline` primitives (requires CUDA 11+).

### Basic Pattern

1.  **Acquire**: Reserve space in the pipeline.
2.  **Copy**: Issue `cuda::memcpy_async`.
3.  **Commit**: Mark the batch of copies.
4.  **Wait**: Block until the copies are ready.
5.  **Release**: Signal that data has been consumed.

**Code Example:** [`AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh)

```cpp
#include <cuda/pipeline>

__global__ void async_kernel(int* global, int* shared) {
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // 1. Acquire
    pipe.producer_acquire();

    // 2. Async Copy
    cuda::memcpy_async(shared, global, sizeof(int), pipe);

    // 3. Commit
    pipe.producer_commit();

    // 4. Wait
    pipe.consumer_wait();

    // ... Use data in shared ...

    // 5. Release
    pipe.consumer_release();
}
```

---

## 3. Performance Benefits

1.  **Reduced Register Pressure**: Since data doesn't sit in registers, you can achieve higher occupancy.
2.  **Latency Hiding**: You can issue multiple loads and then wait for them all at once (or in stages).
3.  **Bandwidth**: The hardware is optimized for these bulk transfers.

---

## 4. Multi-Stage Pipelines

For maximum performance (e.g., in GEMM kernels), a multi-stage pipeline is used:
*   **Stage N**: Loading Global -> Shared
*   **Stage N-1**: Loading Shared -> Register (for Compute)
*   **Stage N-2**: Computing

`cp.async` is critical for the "Global -> Shared" stage to run concurrently with the "Compute" stage.

---

**Next:** [Back to Memory Hierarchy Index](1_cuda_memory_hierarchy.md)
