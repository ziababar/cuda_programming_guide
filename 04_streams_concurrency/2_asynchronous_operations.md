# Asynchronous Operations

Asynchronous execution is the cornerstone of high-performance GPU programming, enabling overlapped computation, memory transfer concurrency, and sophisticated pipeline orchestration.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Previous: [Stream Fundamentals](1_stream_fundamentals.md)**

---

## Compute-Transfer Overlap

The ability to overlap computation with memory transfers is one of the most powerful features of CUDA streams, often yielding 2-4x throughput improvements.

### Key Concepts
*   **Engine Concurrency**: GPUs typically have one or more copy engines (DMA) and compute engines (SMs) that can operate simultaneously.
*   **Pinned Memory**: Essential for asynchronous transfers (see [Memory Transfer](3_memory_transfer.md)).
*   **Depth-First vs Breadth-First**:
    *   *Breadth-First*: Launch all copies, then all kernels. (May serialize).
    *   *Depth-First*: Launch Copy A, Kernel A, Copy B, Kernel B. (Promotes overlap).

### Example: Basic Overlap

```cpp
// Method 2: Chunked with overlap
for (int chunk = 0; chunk < num_streams; chunk++) {
    int offset = chunk * chunk_size;
    cudaStream_t stream = streams[chunk];

    // Async copy input chunk
    cudaMemcpyAsync(&d_input[offset], &h_input[offset],
                   chunk_size * sizeof(float),
                   cudaMemcpyHostToDevice, stream);

    // Process chunk
    complex_processing_kernel<<<(chunk_size+255)/256, 256, 0, stream>>>(
        &d_input[offset], &d_output[offset], chunk_size);

    // Async copy output chunk
    cudaMemcpyAsync(&h_output[offset], &d_output[offset],
                   chunk_size * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
}
```

## Stream Synchronization Mechanisms

Synchronizing streams is crucial for ensuring data integrity and coordinating dependencies.

### Synchronization Primitives
1.  `cudaStreamSynchronize(stream)`: Host waits for a specific stream.
2.  `cudaDeviceSynchronize()`: Host waits for all streams (heavyweight).
3.  `cudaStreamWaitEvent(stream, event)`: Make a stream wait for an event (GPU-side sync, no host blocking).

### StreamSynchronizer Implementation

The `StreamSynchronizer` class provides patterns for:
*   Barrier synchronization
*   Producer-consumer sync
*   Fork-join patterns
*   Pipeline stage synchronization

> **Full Implementation**: [`src/04_streams_concurrency/StreamSynchronizer.cuh`](../src/04_streams_concurrency/StreamSynchronizer.cuh)

```cpp
#include "../src/04_streams_concurrency/StreamSynchronizer.cuh"

void demonstrate_sync_patterns() {
    StreamSynchronizer sync(4);

    // Example: Fork-Join
    std::vector<int> workers = {1, 2, 3};
    sync.fork_join_pattern(workers, 0);
}
```

## Dynamic Stream Management

For workloads with varying intensity, allocating streams dynamically can improve resource utilization.

### AdaptiveStreamManager Implementation

The `AdaptiveStreamManager` handles:
*   Dynamic creation/destruction of streams
*   Utilization tracking
*   Load-based scaling

> **Full Implementation**: [`src/04_streams_concurrency/AdaptiveStreamManager.cuh`](../src/04_streams_concurrency/AdaptiveStreamManager.cuh)

```cpp
#include "../src/04_streams_concurrency/AdaptiveStreamManager.cuh"

void demonstrate_adaptive_streams() {
    AdaptiveStreamManager manager(4, 0.8f);

    // Get a stream based on current load
    int stream_id = manager.acquire_stream();
    cudaStream_t stream = manager.get_cuda_stream(stream_id);

    // Use stream...

    manager.release_stream(stream_id);
    manager.optimize_stream_count(); // Scale up/down
}
```
