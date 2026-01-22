# Asynchronous Memory Copy (cp.async)

Asynchronous Memory Copy (`cp.async`) is a feature introduced in NVIDIA Ampere Architecture (Compute Capability 8.0) that allows data to be copied from global memory directly to shared memory, bypassing the register file.

**Previous: [Memory Debugging](6_memory_debugging.md)** | **Next: [Synchronization](../03_synchronization/1_synchronization_basics.md)**

---

## **The Bottleneck**

In traditional CUDA architectures (pre-Ampere), loading data from global memory to shared memory required intermediate registers:

1.  **Load**: Global Memory $\to$ Register
2.  **Store**: Register $\to$ Shared Memory

This consumes register bandwidth and increases register pressure, potentially limiting occupancy.

## **The Solution: `cp.async`**

`cp.async` (exposed via `cuda::memcpy_async` in C++) initiates a copy operation that is performed by the Direct Memory Access (DMA) engine.

**Benefits:**
1.  **Bypasses Registers**: Data moves directly from L2 Cache/Global Memory to Shared Memory.
2.  **Hides Latency**: The thread can continue executing other instructions (e.g., math) while the copy is in flight.
3.  **Reduces Register Pressure**: Frees up registers for computation.

---

## **Pipeline Pattern**

To maximize performance, `cp.async` is used in a **Software Pipeline**. While the GPU computes on batch $N$, it asynchronously fetches batch $N+1$.

### **Stages**

1.  **Acquire**: Reserve resources for the copy.
2.  **Commit**: Issue the asynchronous copy command.
3.  **Wait**: Block until the copy is complete (or a specific stage is complete).
4.  **Release**: Mark the data as consumed.

### **C++ Interface (`cuda::pipeline`)**

The `cuda::pipeline` class (C++20) provides a high-level interface for this pattern.

```cpp
#include <cuda/pipeline>

__global__ void pipeline_example(int* global, int* shared) {
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // 1. Acquire
    pipe.producer_acquire();

    // 2. Async Copy (Issue)
    cuda::memcpy_async(shared, global, sizeof(int), pipe);

    // 3. Commit (finalize the batch of copies)
    pipe.producer_commit();

    // ... Perform independent work here ...

    // 4. Wait (block until copy finishes)
    pipe.consumer_wait();

    // Access 'shared' data safely
    int val = *shared;

    // 5. Release
    pipe.consumer_release();
}
```

---

## **Double Buffering Example**

See `src/02_memory_hierarchy/AsyncCopyDemo.cuh` for a complete double-buffered implementation.

Double buffering splits shared memory into two halves. While the kernel computes on Buffer A, it loads the next chunk of data into Buffer B.

| Time Step | Buffer A | Buffer B |
| :--- | :--- | :--- |
| **0** | Loading (Async) | - |
| **1** | Computing | Loading (Async) |
| **2** | Loading (Async) | Computing |

```cpp
// Simplified Logic
for (int i = 0; i < N; ++i) {
    int curr = i % 2;
    int next = (i + 1) % 2;

    // Issue Load for NEXT
    if (has_next) {
        pipe.producer_acquire();
        cuda::memcpy_async(s_buff[next], global_ptr + offset, size, pipe);
        pipe.producer_commit();
    }

    // Wait for CURRENT
    pipe.consumer_wait();

    // Compute on CURRENT
    compute(s_buff[curr]);

    // Release CURRENT
    pipe.consumer_release();
}
```

## **Requirements**

-   **Hardware**: Compute Capability 8.0+ (NVIDIA Ampere, Hopper, etc.)
-   **Compiler**: `nvcc` with C++ standard support (typically C++17 or C++20 for `<cuda/pipeline>`).
-   **Header**: `#include <cuda/pipeline>`

## **Related Guides**

-   **[Global Memory](2_global_memory.md)** - Understanding memory transactions.
-   **[Shared Memory](3_shared_memory.md)** - Optimizing bank conflicts.
-   **[Synchronization](../03_synchronization/1_synchronization_basics.md)** - General synchronization concepts.
