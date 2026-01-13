# Asynchronous Memory Copy (`cp.async`)

Introduced in the NVIDIA Ampere Architecture (Compute Capability 8.0), **Asynchronous Memory Copy** (`cp.async` or `cuda::memcpy_async`) allows data to be moved from Global Memory directly to Shared Memory without using intermediate registers.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Next: [Synchronization](../03_synchronization/1_synchronization_basics.md)**

---

## **Overview**

Traditional memory loads follow a "Load -> Register -> Store" pattern. This consumes register bandwidth and blocks the execution thread until the load completes (or uses a lot of registers to hide latency).

`cp.async` offloads the copy operation to a dedicated hardware unit (the **Copy Engine**), allowing the Compute SMs to continue executing instructions or issue more copies.

### **Benefits**
1.  **Register Pressure Reduction**: Bypasses the register file (Global -> Shared direct path).
2.  **Latency Hiding**: Can overlap copy with compute using a multi-stage pipeline.
3.  **Bandwidth Efficiency**: Optimized for bulk transfers.

---

## **The `cuda::pipeline` Interface**

The C++ interface (cuda < 12) or `<cuda/pipeline>` (C++20 style) provides a robust way to manage asynchronous copies.

### **Pipeline Stages**
A typical pipeline involves:
1.  **Acquire**: Reserve a stage in the pipeline.
2.  **Issue**: Dispatch async copy commands.
3.  **Commit**: Mark the end of the batch of commands.
4.  **Wait**: Block until a specific stage is complete.
5.  **Release**: Free the stage.

---

## **Implementation Example**

See [`src/02_memory_hierarchy/AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh) for a complete, compilable example.

### **Kernel Code Snippet**

```cpp
#include <cuda/pipeline>

__global__ void pipeline_kernel(float* global_in, float* global_out) {
    extern __shared__ float smem[];
    // Double buffering pointers
    float* buf[2] = { smem, smem + blockDim.x };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Prologue: Load first tile
    pipe.producer_acquire();
    cuda::memcpy_async(&buf[0][threadIdx.x], &global_in[threadIdx.x], sizeof(float), pipe);
    pipe.producer_commit();

    // Main Loop
    for (int i = 0; i < STEPS; ++i) {
        // Pre-fetch Next Tile (into buf[(i+1)%2])
        pipe.producer_acquire();
        cuda::memcpy_async(&buf[(i+1)%2][threadIdx.x], &global_in[next_idx], sizeof(float), pipe);
        pipe.producer_commit();

        // Wait for Current Tile (buf[i%2])
        pipe.consumer_wait();

        // Compute on Current Tile
        compute(buf[i%2]);

        // Release
        pipe.consumer_release();
    }
}
```

---

## **Hardware Requirements**

- **Compute Capability**: 8.0 or higher (Ampere, Hopper, Blackwell).
- **Compilation**: Must compile with `-arch=sm_80` or higher.

## **Best Practices**

1.  **Double Buffering**: Always use at least 2 buffers (Current, Next) to maximize overlap.
2.  **Wait Strategy**: Use `wait_prior<N>` where N is the number of pending stages you want to keep.
3.  **Pinned Memory**: Host memory should be pinned (`cudaHostAlloc`) if you are copying Host->Device->Shared in a streaming scenario, though `cp.async` is specifically Global->Shared.

---

**[Back to Memory Hierarchy Index](1_cuda_memory_hierarchy.md)**
