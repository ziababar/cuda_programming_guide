# Asynchronous Memory Copy (`cp.async`)

Introduced in CUDA Compute Capability 8.0 (Ampere), `cp.async` allows copying data from global memory directly to shared memory asynchronously, bypassing the register file. This enables efficient overlapping of compute and data transfer phases in a pipeline.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Next: [Synchronization](../03_synchronization/1_synchronization_basics.md)**

---

## **Why `cp.async`?**

In traditional copy operations:
1.  Thread reads from Global Memory into a Register.
2.  Thread writes from Register into Shared Memory.
3.  Thread is blocked or uses instruction-level parallelism (ILP) to hide latency.

With `cp.async`:
1.  Thread issues a "copy command".
2.  DMA engine moves data **directly** from L2 Cache/Global Memory to Shared Memory.
3.  Threads are free to perform other work (compute).
4.  Threads wait (barrier) only when the data is needed.

**Benefits:**
*   **Reduced Register Pressure**: Data doesn't pass through registers.
*   **Latency Hiding**: Perfect for pipelining (e.g., loading the next tile while computing the current one).
*   **Bandwidth Efficiency**: optimized for bulk transfers.

---

## **The `cuda::pipeline` API**

Modern CUDA C++ (CUDA 11+) provides the `<cuda/pipeline>` header (part of libcu++) to manage these asynchronous operations safely.

### **Key Concepts**

1.  **`cuda::pipeline`**: Object coordinating the transfer.
2.  **`producer_acquire()`**: Signal that we are preparing to issue copy commands.
3.  **`memcpy_async()`**: Issue the asynchronous copy.
4.  **`producer_commit()`**: Signal that all copies for this "stage" are issued.
5.  **`consumer_wait()`**: Block the thread until the data is ready.
6.  **`consumer_release()`**: Signal that we are done using the data.

### **Pipeline Stages**

Pipelines are often described by the number of "stages" (buffers) they use.
*   **Multi-Stage**: N buffers. While computing stage `k`, we are loading stage `k+1`, `k+2`, etc.

---

## **Implementation Pattern**

Below is a standard pattern for a loop using a pipeline.

### **Single-Stage (Look-Ahead)**

```cpp
#include <cuda/pipeline>

__global__ void pipeline_kernel(float* global_in, float* global_out) {
    extern __shared__ float shared_mem[];
    float* buffer = shared_mem;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Loop
    for (int i = 0; i < N; ++i) {
        // 1. Issue Copy
        pipe.producer_acquire();
        cuda::memcpy_async(&buffer[threadIdx.x], &global_in[idx], 4, pipe);
        pipe.producer_commit();

        // 2. Wait for Copy
        pipe.consumer_wait();

        // 3. Compute
        float val = buffer[threadIdx.x];
        // ... heavy math ...

        // 4. Release
        pipe.consumer_release();
    }
}
```

### **Double Buffering (2-Stage Pipeline)**

To truly hide latency, we need at least two buffers: compute on one while loading the other.

```cpp
#include <cuda/pipeline>

__global__ void double_buffer_kernel(float* in, float* out) {
    extern __shared__ float smem[];
    float* buffers[2];
    buffers[0] = smem;
    buffers[1] = smem + blockDim.x;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Prologue: Load first batch
    pipe.producer_acquire();
    cuda::memcpy_async(&buffers[0][threadIdx.x], &in[base_idx], 4, pipe);
    pipe.producer_commit();

    // Main Loop
    for (int i = 0; i < num_batches; ++i) {
        int curr = i % 2;
        int next = (i + 1) % 2;

        // Issue Load for Next Batch (if exists)
        if (i < num_batches - 1) {
            pipe.producer_acquire();
            cuda::memcpy_async(&buffers[next][threadIdx.x], &in[next_idx], 4, pipe);
            pipe.producer_commit();
        }

        // Wait for Current Batch
        pipe.consumer_wait();

        // Compute on Current Batch
        // ... compute using buffers[curr] ...

        // Release Current Batch
        pipe.consumer_release();
    }
}
```

---

## **Requirements**

*   **Hardware**: Compute Capability 8.0+ (Ampere, Hopper, etc.)
*   **Software**: CUDA Toolkit 11.0+
*   **Compilation**: `nvcc -arch=sm_80 ...`

## **Code Example**

For a complete, compilable example of a double-buffered pipeline, see `src/02_memory_hierarchy/AsyncCopyDemo.cuh`.

```cpp
// Snippet from AsyncCopyDemo.cuh
pipe.producer_acquire();
cuda::memcpy_async(&next_buffer[threadIdx.x], &global_in[global_idx], sizeof(float), pipe);
pipe.producer_commit();

// ... compute concurrent with copy ...

pipe.consumer_wait();
// use data
pipe.consumer_release();
```

---

## **Key Takeaways**

1.  **Bypass Registers**: `cp.async` moves data directly to shared memory.
2.  **Overlap**: Use pipelines (multi-stage) to overlap memory transfer with computation.
3.  **Alignment**: Ensure proper alignment (16 bytes is ideal) for maximum throughput.
4.  **Hardware Support**: Check for SM 8.0+ before using.
