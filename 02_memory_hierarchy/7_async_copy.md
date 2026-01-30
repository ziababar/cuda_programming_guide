# Asynchronous Memory Copy (`cp.async`)

Asynchronous Memory Copy, introduced in the NVIDIA Ampere Architecture (Compute Capability 8.0), allows threads to initiate a copy from Global Memory directly to Shared Memory without using intermediate registers.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **[Back to Index](1_cuda_memory_hierarchy.md)**

---

## Why `cp.async`?

### Traditional Copy (Pre-Ampere)
1.  **Load**: Global Memory $\to$ Register (Stall)
2.  **Store**: Register $\to$ Shared Memory
3.  **Sync**: `__syncthreads()`

This approach consumes register file bandwidth and blocks execution until the data arrives (unless carefully prefetched).

### Asynchronous Copy (Ampere+)
1.  **Issue**: Thread issues `cp.async` instruction.
2.  **Background**: Data moves Global $\to$ Shared Memory (Bypassing RF).
3.  **Compute**: Thread continues execution (e.g., math on previous tile).
4.  **Wait**: `pipeline.wait()` ensures data has arrived.

**Benefits:**
- **Reduced Register Pressure**: Data doesn't pass through registers.
- **Latency Hiding**: Perfect overlap of Compute and Memory Copy.
- **Bandwidth Efficiency**: Better utilization of the memory subsystem.

---

## The `cuda::pipeline` API

CUDA C++ provides the `<cuda/pipeline>` header to manage asynchronous copies.

### Key Primitives

| Function | Description |
|----------|-------------|
| `cuda::make_pipeline()` | Creates a pipeline object. |
| `pipe.producer_acquire()` | Reserves a stage in the pipeline for new data. |
| `cuda::memcpy_async` | Issues the copy operation (non-blocking). |
| `pipe.producer_commit()` | Signals that all copies for the current stage are issued. |
| `pipe.consumer_wait()` | Blocks until the oldest active stage is ready. |
| `pipe.consumer_release()` | Marks the stage as consumed, freeing it for reuse. |

---

## Implementation Pattern

The most common pattern is a multi-stage pipeline (Double or Triple Buffering).

### Code Example

See `src/02_memory_hierarchy/AsyncCopyDemo.cuh` for a complete implementation.

```cpp
#include <cuda/pipeline>

template <int BLOCK_SIZE>
__global__ void pipeline_kernel(float* d_out, float* d_in) {
    extern __shared__ float smem[];
    float* stage0 = smem;
    float* stage1 = smem + BLOCK_SIZE;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Prologue: Load first tile
    pipe.producer_acquire();
    cuda::memcpy_async(stage0, d_in, size, pipe);
    pipe.producer_commit();

    // Loop
    for (int i = 0; i < N; ++i) {
        // 1. Issue Next Copy
        pipe.producer_acquire();
        cuda::memcpy_async(stage1, d_in + next_offset, size, pipe);
        pipe.producer_commit();

        // 2. Wait for Current Copy
        pipe.consumer_wait();

        // 3. Compute (on stage0)
        compute(stage0);

        // 4. Release
        pipe.consumer_release();

        // Swap pointers
        swap(stage0, stage1);
    }
}
```

---

## Requirements

-   **Hardware**: NVIDIA Ampere (SM 8.0) or newer (e.g., A100, RTX 30/40 series).
-   **Compiler**: CUDA 11.0+.
-   **Flag**: Compile with `-arch=sm_80` or higher.

```bash
nvcc -arch=sm_80 -o async_app app.cu
```

## Performance Tips

1.  **Batch Copies**: Issue multiple `memcpy_async` calls between `acquire` and `commit`.
2.  **Unroll Loops**: Help the compiler schedule instructions better.
3.  **Shared Memory Layout**: Ensure destination addresses in shared memory avoid bank conflicts when read later.
