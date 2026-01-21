# Asynchronous Memory Copy (`cp.async`)

Asynchronous Memory Copy, introduced in NVIDIA Ampere architecture (Compute Capability 8.0), allows data to be moved from Global Memory directly to Shared Memory without using intermediate registers.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Next: [Synchronization Basics](../03_synchronization/1_synchronization_basics.md)**

---

## Why Asynchronous Copy?

In older architectures, loading data from Global Memory to Shared Memory required a two-step process:
1.  **Global → Register**: Data is fetched into a thread's register.
2.  **Register → Shared**: Data is written from the register to shared memory.

This approach wastes register bandwidth and consumes register space.

**`cp.async` (Async Copy)** creates a direct path:
*   **Global → Shared**: The DMA engine handles the transfer.
*   **Non-blocking**: The thread issues the copy command and can continue executing other instructions (e.g., math) while data arrives.
*   **Register Bypass**: Frees up registers for computation.

---

## The `cuda::pipeline` Interface

The modern C++ interface for `cp.async` is provided by the `<cuda/pipeline>` header. It manages the complexity of tracking asynchronous operations.

### Key Concepts

1.  **Stages**: A set of memory operations issued together.
2.  **Acquire**: Reserve resources for a new stage (producer).
3.  **Commit**: Mark the end of a batch of copy commands (producer).
4.  **Wait**: Block until a specific stage is complete (consumer).
5.  **Release**: Signal that the data has been consumed and the stage resources can be reused (consumer).

---

## Code Example: Multi-Stage Pipeline

The following example demonstrates a double-buffered pipeline where we load the *next* batch of data while processing the *current* batch.

**Source:** [`src/02_memory_hierarchy/AsyncCopyDemo.cuh`](../src/02_memory_hierarchy/AsyncCopyDemo.cuh)

```cpp
#include <cuda/pipeline>

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void async_copy_pipeline_kernel(float* d_in, float* d_out, int N) {
    // Double buffering (2 stages)
    __shared__ float smem_buffer[2][BLOCK_SIZE * TILE_SIZE];

    auto block = cg::this_thread_block();
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Loop over batches
    for (int batch = 0; batch < num_batches; ++batch) {
        // 1. PRODUCER: Issue Copy for NEXT batch
        pipe.producer_acquire();
        // Issue async copy commands
        cuda::memcpy_async(dst, src, size, pipe);
        pipe.producer_commit();

        // 2. CONSUMER: Wait for CURRENT batch
        pipe.consumer_wait();

        // ... Perform Compute on smem_buffer[current] ...
        block.sync(); // Ensure all threads finished reading

        // 3. Release stage
        pipe.consumer_release();
    }
}
```

### Explanation

1.  **`cuda::memcpy_async`**: This function generates the `cp.async` PTX instruction. It tells the hardware to move bytes from global to shared memory.
2.  **`pipe.producer_acquire()`**: Prepares the pipeline to accept new commands.
3.  **`pipe.producer_commit()`**: Groups all preceding `memcpy_async` calls into a single "stage".
4.  **`pipe.consumer_wait()`**: Stalls the thread until the oldest active stage is finished. In a loop, this creates the overlap effect—we are waiting for data requested in the *previous* iteration while the hardware is already fetching data for the *next* iteration.

---

## Hardware Requirements

*   **Compute Capability**: 8.0 or higher (Ampere, Hopper, Ada Lovelace).
*   **Compilation**: Must compile with `-arch=sm_80` or higher.

If you try to run this on older hardware (Volta, Pascal), it will fail or fall back to synchronous copies depending on the toolkit version, but typically `cuda::pipeline` requires sm_80+ for the hardware acceleration.

## Best Practices

*   **128-Byte Alignment**: For maximum performance, ensure that global memory addresses and shared memory pointers are 128-byte aligned. This allows the hardware to issue the widest possible transactions.
*   **Warp Efficiency**: While `memcpy_async` is per-thread, the hardware coalesces these requests. Ensure that the warp accesses contiguous global memory to form 128-byte transactions.
*   **Shared Memory Banks**: Be mindful of bank conflicts when *reading* the data in the compute phase (Consumer), just like with normal shared memory. The Async Copy itself writes to shared memory efficiently avoiding bank conflicts if the pattern is linear.
