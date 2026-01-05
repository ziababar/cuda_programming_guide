# Stream Fundamentals

CUDA streams are the backbone of high-performance GPU programming, enabling asynchronous execution, memory transfer overlap, and sophisticated pipeline orchestration. Understanding streams deeply is essential for achieving optimal GPU utilization and building scalable parallel applications.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Related: [Memory Hierarchy](../02_memory_hierarchy/1_cuda_memory_hierarchy.md)**

---

## Stream Types and Properties

CUDA streams represent ordered sequences of GPU operations that execute asynchronously with respect to the host and other streams.

### Stream Hierarchy and Characteristics

1.  **Default Stream (Stream 0)**:
    *   Synchronous with the host (blocking behavior).
    *   Implicitly synchronizes with all other streams (legacy behavior) or acts independently (per-thread default stream).
    *   Used when no explicit stream is specified.

2.  **Explicit Streams**:
    *   Asynchronous with the host.
    *   Can execute concurrently with other streams.
    *   Enable overlap of computation and memory transfers.

### Code Example: Stream Fundamentals

```cpp
// Comprehensive stream type demonstration
void demonstrate_stream_fundamentals() {
    printf("=== CUDA Stream Fundamentals ===\n");

    // 1. Default Stream (Stream 0) - Synchronous Behavior
    printf("1. Default Stream Characteristics:\n");
    printf("   - Synchronous with host\n");
    printf("   - Blocks other streams until completion\n");

    // ... (Sequential execution logic) ...

    // 2. Explicit Streams - Asynchronous Behavior
    printf("2. Explicit Stream Characteristics:\n");
    printf("   - Asynchronous with host\n");
    printf("   - Can execute concurrently with other streams\n");

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // These can execute concurrently
    // cudaMemsetAsync(d_data1, 0, size, stream1);
    // simple_kernel<<<256, 256, 0, stream1>>>(d_data1, 1024);

    // ...
}
```

## Stream Execution Model

Operations within a single stream are serialized (FIFO). Operations in different streams can be interleaved or run concurrently by the GPU scheduler.

*   **FIFO Ordering**: `Kernel A` -> `Kernel B` submitted to the same stream are guaranteed to execute in that order.
*   **Inter-Stream Concurrency**: `Stream 1 (Kernel A)` and `Stream 2 (Kernel B)` have no ordering guarantees relative to each other and may run in parallel if resources allow.

## Advanced Stream Management

For production applications, managing a pool of streams is often necessary. We provide a robust `StreamManager` class to handle stream creation, prioritization, and lifecycle management.

### StreamManager Implementation

The `StreamManager` class provides:
*   Pool-based stream management
*   Priority support
*   Round-robin or load-based allocation
*   RAII-based cleanup

> **Full Implementation**: [`src/04_streams_concurrency/StreamManager.cuh`](../src/04_streams_concurrency/StreamManager.cuh)

```cpp
#include "../src/04_streams_concurrency/StreamManager.cuh"

void demonstrate_stream_manager() {
    printf("=== Stream Manager Demo ===\n");

    // Create manager with priority streams and custom tags
    std::vector<std::string> tags = {"MemoryOps", "Compute", "HighPrio", "Background"};
    StreamManager manager(4, true, tags);

    // Assign workloads to appropriate streams
    cudaStream_t memory_stream = manager.get_stream_for_workload("memory_intensive");
    cudaStream_t compute_stream = manager.get_stream_for_workload("compute_intensive");

    // ...
}
```
