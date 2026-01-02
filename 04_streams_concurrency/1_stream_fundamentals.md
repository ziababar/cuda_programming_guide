# CUDA Stream Fundamentals

CUDA streams represent ordered sequences of GPU operations that execute asynchronously with respect to the host and other streams, enabling sophisticated concurrency patterns.

## Stream Types and Properties

### Stream Hierarchy and Characteristics

1. **Default Stream (Stream 0)**:
   - Synchronous with host.
   - Blocks other streams until completion.
   - Used when no explicit stream is specified.

2. **Explicit Streams (Asynchronous)**:
   - Asynchronous with host.
   - Can execute concurrently with other streams.
   - Enable overlap of computation and data transfer.

```cpp
// Example: Demonstrating stream fundamentals
// Full implementation available in src/04_streams_concurrency/stream_manager.cuh

void demonstrate_stream_fundamentals() {
    // ... setup code ...

    // Explicit streams allow concurrent execution
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Operations in different streams can overlap
    cudaMemsetAsync(d_data1, 0, size, stream1);
    cudaMemsetAsync(d_data2, 1, size, stream2);

    // ... cleanup ...
}
```

### Stream Execution Model

Streams follow a strict FIFO (First-In-First-Out) ordering for operations within the same stream. However, operations in different streams can execute concurrently, limited only by hardware resources and dependencies.

- **Intra-Stream**: Sequential execution.
- **Inter-Stream**: Concurrent execution (where possible).

For a complete demonstration of the stream execution model, including concurrent kernel execution and memory transfers, refer to the source code examples.

## Stream Management Patterns

Managing a large number of streams efficiently is crucial for complex applications. A `StreamManager` class can help handle stream creation, reuse, and priority management.

### Advanced Stream Management

A `StreamManager` typically handles:
- **Pooling**: reusing streams to avoid creation/destruction overhead.
- **Prioritization**: managing streams with different priorities (e.g., High, Low).
- **Load Balancing**: assigning work to available streams.

> **Note**: A comprehensive `StreamManager` implementation is provided in [`src/04_streams_concurrency/stream_manager.cuh`](../src/04_streams_concurrency/stream_manager.cuh). It includes features like round-robin allocation, availability checking, and priority-based selection.

```cpp
// Usage example of StreamManager
#include "../src/04_streams_concurrency/stream_manager.cuh"

void demonstrate_stream_manager() {
    // Create manager with priority streams and custom tags
    std::vector<std::string> tags = {"MemoryOps", "Compute", "HighPrio", "Background"};
    StreamManager manager(4, true, tags);

    // Assign workloads to appropriate streams
    cudaStream_t compute_stream = manager.get_stream_for_workload("compute_intensive");

    // Launch operations
    // ...
}
```
