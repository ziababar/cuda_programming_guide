# Asynchronous Memory Copy (`cp.async`)

Introduced in CUDA 11 (Ampere Architecture, Compute Capability 8.0+), Asynchronous Memory Copy (`cp.async`) is a powerful feature that allows data to be copied from Global Memory directly to Shared Memory, **bypassing the register file**.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Next: [Synchronization](../03_synchronization/1_synchronization_basics.md)**

---

## **Why `cp.async`?**

In traditional CUDA programming, loading data from global memory to shared memory involves two steps:
1.  **Load to Register**: `LDG` instruction (Global -> Register)
2.  **Store to Shared**: `STS` instruction (Register -> Shared)

This approach consumes register bandwidth and occupies registers that could be used for computation.

**`cp.async` (Async Copy)** streamlines this:
-   **Direct Path**: Data moves from L2 Cache/Global Memory -> Shared Memory.
-   **Non-Blocking**: The copy instruction is issued asynchronously. The thread can continue executing other independent instructions (e.g., math) while the data is in flight.
-   **Register Relief**: Does not use general-purpose registers for the data transfer.

---

## **The `cuda::pipeline` Interface**

CUDA C++ provides the `cuda::pipeline` primitive (in `<cuda/pipeline>`) to manage these asynchronous operations. It follows a multi-stage process:

1.  **Acquire**: Request access to the pipeline for issuing commands.
2.  **Issue**: Submit asynchronous copy commands (e.g., `cuda::memcpy_async`).
3.  **Commit**: Mark the end of a batch of commands.
4.  **Wait**: Block execution until the specified batch is complete.
5.  **Release**: Release the pipeline resources.

### **Code Example**

Below is a simplified example of using `cp.async` to load a value, wait for it, and then process it.

> **Note**: This feature requires `Compute Capability >= 8.0`.

```cpp
#include <cuda/pipeline>

__global__ void async_copy_kernel(float* d_out, const float* d_in, int N) {
    extern __shared__ float s_data[];

    // Create a pipeline object managed by this thread
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        // 1. Acquire: Reserve resources for the copy
        pipe.producer_acquire();

        // 2. Issue: Async Copy from Global -> Shared
        // cuda::memcpy_async(dst_shared, src_global, size, pipeline)
        cuda::memcpy_async(&s_data[tid], &d_in[idx], sizeof(float), pipe);

        // 3. Commit: Finalize the batch of copies
        pipe.producer_commit();

        // ... We could do independent math here ...

        // 4. Wait: Wait for the copy to complete
        pipe.consumer_wait();

        // 5. Compute: Data is now safe to read from shared memory
        float val = s_data[tid] * 2.0f;

        // 6. Release: Done with this batch
        pipe.consumer_release();

        d_out[idx] = val;
    }
}
```

For a complete, compilable example, see **`src/02_memory_hierarchy/AsyncCopyDemo.cuh`**.

---

## **Pipeline Patterns**

Real-world usage often involves loops where we overlap the **computation of batch `i`** with the **copy of batch `i+1`**.

### **Double Buffering with Pipelines**

```cpp
// Pseudo-code for a pipelined loop
pipe.producer_acquire();
cuda::memcpy_async(buffer[0], src[0], size, pipe);
pipe.producer_commit();

for (int i = 0; i < N; ++i) {
    // Issue NEXT copy (i+1)
    if (i < N - 1) {
       pipe.producer_acquire();
       cuda::memcpy_async(buffer[(i+1)%2], src[i+1], size, pipe);
       pipe.producer_commit();
    }

    // Wait for CURRENT copy (i)
    pipe.consumer_wait();

    // Compute on CURRENT buffer
    compute(buffer[i%2]);

    // Release CURRENT buffer
    pipe.consumer_release();
}
```

---

## **Benefits**

1.  **Latency Hiding**: Overlap memory transfers with computation.
2.  **Reduced Register Pressure**: Frees up registers for complex math (e.g., tensor cores).
3.  **Bandwidth Efficiency**: Maximizes memory throughput by keeping the memory bus busy.

## **Hardware Requirements**

-   **Architecture**: NVIDIA Ampere (SM 80) or newer.
-   **Compiler**: `nvcc` with `-arch=sm_80` or higher.
