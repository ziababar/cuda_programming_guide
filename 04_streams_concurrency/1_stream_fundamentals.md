# Stream Fundamentals

CUDA streams represent ordered sequences of GPU operations that execute asynchronously with respect to the host and other streams, enabling sophisticated concurrency patterns.

**[Back to Index](1_cuda_streams_concurrency.md)**

---

## Stream Types and Properties

### Stream Hierarchy and Characteristics

```cpp
// Comprehensive stream type demonstration
// See src/04_streams_concurrency/1_stream_fundamentals.cuh for full implementation
void demonstrate_stream_fundamentals();
```

The default stream (Stream 0) has synchronous behavior with respect to the host and blocks other streams until completion. Explicit streams are asynchronous and can execute concurrently.

### Stream Execution Model

```cpp
// Demonstrate FIFO ordering and inter-stream concurrency
// See src/04_streams_concurrency/1_stream_fundamentals.cuh for full implementation
void demonstrate_stream_execution_model();
```

Key concepts:
1.  **FIFO Ordering Within Streams**: Operations within each stream execute in submission order.
2.  **Inter-Stream Concurrency**: Different streams can execute concurrently. The GPU scheduler interleaves stream operations.
3.  **Synchronization Points**: `cudaStreamSynchronize()` waits for a specific stream, `cudaDeviceSynchronize()` waits for all streams.

## Stream Management Patterns

### Advanced Stream Management

A `StreamManager` class can help manage streams, priorities, and workload assignment.

```cpp
// Sophisticated stream management for production applications
// See src/04_streams_concurrency/1_stream_fundamentals.cuh for full implementation
class StreamManager {
    // ...
};
```

## Nsight Debugging Tips

- Use **Nsight Systems** to visualize:
  - Stream timelines
  - Overlap of memcopy and kernels
- Identify serialization caused by:
  - Shared resources
  - Host sync calls (`cudaDeviceSynchronize()`)
