# Asynchronous Operations

Asynchronous execution is the cornerstone of high-performance GPU programming, enabling overlapped computation, memory transfer concurrency, and sophisticated pipeline orchestration.

## Compute-Transfer Overlap

The ability to overlap computation with memory transfers is one of the most powerful features of CUDA streams, often yielding 2-4x throughput improvements. This is typically achieved by dividing data into chunks and processing them in a pipeline fashion across multiple streams.

### Basic Overlap Patterns

1. **Sequential (No Overlap)**:
   - Copy H2D -> Kernel -> Copy D2H
   - Total time = Sum of all operations.

2. **Chunked with Overlap**:
   - Split data into N chunks.
   - Use N streams.
   - For each stream: Async Copy H2D -> Async Kernel -> Async Copy D2H.
   - While Stream 0 is computing, Stream 1 can be copying data.

### Advanced Pipeline Processing

A sophisticated pipeline might involve multiple processing stages (e.g., Input Transfer -> Stage 1 -> Stage 2 -> Output Transfer).

```cpp
// Example: Pipeline processing
// See src/04_streams_concurrency/stream_pipeline.cuh for a full implementation

void pipeline_processing_demo(...) {
    // ...
    for (int chunk = 0; chunk < total_chunks; chunk++) {
        int stream_id = chunk % num_streams;
        cudaStream_t stream = streams[stream_id];

        // Pipeline stages
        cudaMemcpyAsync(..., stream); // Input
        stage1_kernel<<<..., stream>>>(...);
        stage2_kernel<<<..., stream>>>(...);
        cudaMemcpyAsync(..., stream); // Output
    }
    // ...
}
```

> **Reference**: For a reusable pipeline implementation, check [`src/04_streams_concurrency/stream_pipeline.cuh`](../src/04_streams_concurrency/stream_pipeline.cuh).

## Stream Synchronization Mechanisms

Synchronization is required to ensure data integrity and coordinate dependencies.

### Common Mechanisms

- **`cudaStreamSynchronize(stream)`**: Blocks host until the stream is idle.
- **`cudaDeviceSynchronize()`**: Blocks host until all streams on the device are idle.
- **`cudaStreamWaitEvent(stream, event)`**: Makes a stream wait for an event (recorded in another stream) without blocking the host. This is key for inter-stream dependencies.

### Comprehensive Synchronization Patterns

The [`src/04_streams_concurrency/stream_sync.cuh`](../src/04_streams_concurrency/stream_sync.cuh) header defines a `StreamSynchronizer` class that demonstrates:
- **Barrier Synchronization**: Sync all streams at a specific point.
- **Producer-Consumer Sync**: Coordinate data hand-off between streams.
- **Fork-Join**: Launch parallel work in multiple streams and wait for all to finish before proceeding.

### Dynamic Stream Management

Adaptive stream allocation can optimize resource usage based on workload. The `AdaptiveStreamManager` in [`src/04_streams_concurrency/adaptive_stream_manager.cuh`](../src/04_streams_concurrency/adaptive_stream_manager.cuh) shows how to:
- Manage a pool of streams.
- Dynamically create/destroy streams based on utilization.
- Track stream status (Busy/Idle).
