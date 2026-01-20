# Asynchronous Memory Copy (`cp.async`)

Introduced in the NVIDIA Ampere architecture (Compute Capability 8.0), Asynchronous Memory Copy (`cp.async`) is a hardware feature that allows copying data from Global Memory directly to Shared Memory without blocking the execution of other instructions. This bypasses the register file, reducing register pressure and enabling efficient software pipelining.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Return to: [Memory Hierarchy](1_cuda_memory_hierarchy.md)**

---

## Why Asynchronous Copy?

In traditional CUDA kernels, loading data from Global to Shared Memory looks like this:

```cpp
// 1. Load from Global to Register
float val = global_ptr[idx];
// 2. Store from Register to Shared
shared_ptr[idx] = val;
// 3. Wait for all threads to finish
__syncthreads();
```

This approach has two downsides:
1.  **Register Pressure**: It uses registers as intermediate storage.
2.  **Latency**: The thread stalls waiting for the load to complete before it can store.

With `cp.async`, the hardware manages the transfer directly:

```cpp
// Initiate copy from Global to Shared (non-blocking)
__pipeline_memcpy_async(&shared_ptr[idx], &global_ptr[idx], sizeof(float));
// Commit the copy
__pipeline_commit();
// Wait for completion (later)
__pipeline_wait_prior(0);
```

---

## Multi-Stage Pipelines

The real power of `cp.async` comes when combining it with **multi-buffering** (pipelining). While one buffer is being processed by the compute units, the next buffer is being pre-fetched from global memory.

### 2-Stage Pipeline Example (Concept)

1.  **Prologue**: Kick off loading Batch 0.
2.  **Loop**:
    *   Kick off loading Batch `i+1`.
    *   Wait for Batch `i` to arrive.
    *   Process Batch `i`.
3.  **Epilogue**: Process the final batch.

---

## Using `cuda::pipeline` (C++ Interface)

The `<cuda/pipeline>` header (CUDA 11+) provides a C++ abstraction for these operations, making them safer and easier to use than raw intrinsics.

**Requirements**:
*   Compute Capability >= 8.0 (Ampere or newer).
*   Compile with `-arch=sm_80` or higher.

### Basic Syntax

```cpp
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void pipeline_kernel(float* global_in, float* global_out, int N) {
    extern __shared__ float shared_mem[];
    float* s_buffer = shared_mem;

    auto block = cg::this_thread_block();
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // ... setup indices ...

    // 1. Submit Copy
    pipe.producer_acquire();
    cuda::memcpy_async(&s_buffer[tid], &global_in[idx], sizeof(float), pipe);
    pipe.producer_commit();

    // 2. Wait for Copy
    pipe.consumer_wait();

    // 3. Use Data
    float val = s_buffer[tid];

    // 4. Release
    pipe.consumer_release();
}
```

---

## Performance Considerations

1.  **Alignment**: `cp.async` works best with 16-byte aligned addresses (e.g., `float4`).
2.  **Warp Coalescing**: Ensure that the pattern of `cp.async` calls across the warp forms a coalesced memory transaction.
3.  **L2 Cache**: The data passes through the L2 cache. Optimizing L2 locality is still important.

## Detailed Code Example

A full, compilable example demonstrating a multi-stage pipeline is available in:
[`src/02_memory_hierarchy/AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh)

## Key Takeaways

*   **Bypass Registers**: `cp.async` moves data Gmem -> Smem, saving registers.
*   **Hide Latency**: Use pipelines to overlap Compute with Memory Transfers.
*   **Hardware Requirement**: Requires Ampere (SM 8.0) or newer.

---

**Next: [Synchronization](../03_synchronization/1_synchronization_basics.md)**
