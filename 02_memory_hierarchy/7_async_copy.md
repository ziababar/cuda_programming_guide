# Asynchronous Memory Copy (`cp.async`)

Introduced in CUDA 11 (Ampere Architecture, Compute Capability 8.0), **Asynchronous Memory Copy** allows threads to initiate loads from Global Memory to Shared Memory that bypass the register file and do not block execution.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Next: [Synchronization](../03_synchronization/1_synchronization_basics.md)**

---

## **Why `cp.async`?**

Traditional memory loads are blocking or consume register bandwidth:
1.  **Registers**: Loading `Global -> Register -> Shared` wastes register bandwidth and capacity.
2.  **Latency**: Threads stall while waiting for global memory.

`cp.async` (exposed via `cuda::memcpy_async` and `cuda::pipeline` in C++) allows:
1.  **Direct Copy**: Data moves `Global -> L2 -> Shared` (bypassing registers).
2.  **Latency Hiding**: Issue copy command $\to$ Do independent work $\to$ Wait for data.
3.  **Pipeline Overlap**: Create multi-stage pipelines (Load Next, Compute Current).

---

## **The Pipeline Pattern**

The most common use case is a multi-stage pipeline using `cuda::pipeline`.

### **Code Example**

See [`AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh) for a complete compilable example.

```cpp
#include <cuda/pipeline>

__global__ void pipeline_kernel(float* global_in, float* global_out) {
    extern __shared__ float smem[]; // Multi-stage buffer

    // Create pipeline object
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // ... loop setup ...

    for (int i = 0; i < STEPS; ++i) {
        // 1. ACQUIRE: Reserve resources for the copy
        pipe.producer_acquire();

        // 2. ISSUE: Async copy (Global -> Shared)
        // Bypasses registers!
        cuda::memcpy_async(&smem[next_stage_idx],
                           &global_in[global_idx],
                           size,
                           pipe);

        // 3. COMMIT: Mark the batch of copies as "in flight"
        pipe.producer_commit();

        // 4. WAIT: Ensure the specific previous stage is ready
        // consumer_wait() blocks until the oldest pending stage is complete.
        pipe.consumer_wait();

        // 5. COMPUTE: Process data in Shared Memory
        compute(smem[current_stage_idx]);

        // 6. RELEASE: Signal that we are done with this stage's buffer
        pipe.consumer_release();
    }
}
```

---

## **Key APIs**

### **`cuda::memcpy_async`**
Initiates the copy.
```cpp
void cuda::memcpy_async(void* dst, const void* src, size_t size, cuda::pipeline& pipe);
```

### **`pipe.producer_commit()`**
Groups all preceding `memcpy_async` calls into a single "batch" or stage.

### **`pipe.consumer_wait()`**
Blocks the thread until the oldest uncompleted stage is ready.
*   The `cuda::pipeline` class maintains a FIFO queue of stages. `consumer_wait()` ensures the head of the queue is finished.

---

## **Performance Impact**

Using `cp.async` can significantly improve performance for memory-bound kernels (like GEMM, Convolution) by:
*   Reducing register pressure (no need to hold loaded values in registers).
*   Perfectly overlapping Compute with Memory access.

**Requirements:**
*   Compute Capability 8.0+ (Ampere, Hopper, etc.).
*   CUDA 11.0+.
