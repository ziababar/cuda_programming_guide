# Asynchronous Memory Copy (`cp.async`)

Introduced in the Ampere architecture (Compute Capability 8.0), **Asynchronous Memory Copy** (`cp.async`) allows threads to issue copy instructions from Global Memory to Shared Memory that bypass the Register File. This significantly reduces register pressure and enables overlap of memory transfer with computation.

**[Back to Memory Hierarchy Index](1_cuda_memory_hierarchy.md)**

---

## Key Concepts

1.  **Direct Copy**: Data moves `Global -> Shared` without passing through thread registers.
2.  **Asynchronous**: The instruction issues a copy command and returns immediately.
3.  **Pipeline Pattern**: Use `cuda::pipeline` (C++20/LibCu++) to manage stages of data arrival.

## Why use `cp.async`?

| Feature | Standard Copy (`gmem -> reg -> smem`) | Async Copy (`cp.async`) |
| :--- | :--- | :--- |
| **Path** | Global -> Register -> Shared | Global -> Shared |
| **Latency Hiding** | Limited by ILP | Hided by pipeline stages |
| **Register Usage** | High (temporary storage) | Zero (for data) |
| **Throughput** | Lower | Near Peak DRAM Bandwidth |

## Implementation with `cuda::pipeline`

The Modern CUDA C++ way to handle async copies is using the `<cuda/pipeline>` header.

### Pipeline Stages

To hide latency effectively, we use multi-stage buffering (e.g., 3 stages):
*   **Stage N**: Loading (Issuing Copy)
*   **Stage N-1**: Arriving (In flight)
*   **Stage N-2**: Computing (Ready)

### Code Example

See `src/02_memory_hierarchy/AsyncCopyDemo.cuh` for a complete, compilable example.

```cpp
#include <cuda/pipeline>

template<int STAGES=3>
__global__ void pipeline_kernel(float* input, float* output, size_t n) {
    extern __shared__ float shared_mem[];
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // ... setup offsets ...

    // Prologue: Fill the pipeline
    for (int s = 0; s < STAGES; ++s) {
        pipe.producer_acquire();
        // cp.async instruction
        cuda::memcpy_async(&shared_mem[s * tile_size + tid],
                         &input[offset + s], sizeof(float), pipe);
        pipe.producer_commit();
    }

    // Main Loop
    for (int i = 0; i < num_batches; ++i) {
        // 1. Wait for the oldest batch to be ready
        pipe.consumer_wait();

        // 2. Compute on shared memory data
        output[idx] = shared_mem[(i % STAGES) * tile_size + tid] * 2.0f;

        // 3. Release the stage
        pipe.consumer_release();

        // 4. Issue next batch to reuse the stage
        pipe.producer_acquire();
        cuda::memcpy_async(&shared_mem[(i % STAGES) * tile_size + tid],
                         &input[next_offset], sizeof(float), pipe);
        pipe.producer_commit();
    }
}
```

## Compilation Requirements

*   **Compute Capability**: 8.0+ (`-arch=sm_80`)
*   **CUDA Toolkit**: 11.0+

## Performance Impact

Using `cp.async` allows the SM to continue executing other warps or independent instructions while data travels from DRAM to L1/Shared Memory. In bandwidth-bound kernels (like GEMM or Stencils), this can result in:
*   **1.2x - 1.5x speedup** over optimized register-based tiling.
*   **Reduced register pressure**, allowing higher occupancy.

---

## Related Guides

*   **Previous**: [Shared Memory](3_shared_memory.md)
*   **Next**: [Streams & Concurrency](../04_streams_concurrency/1_stream_fundamentals.md)
