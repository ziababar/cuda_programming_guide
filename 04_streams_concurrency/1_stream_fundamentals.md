# Stream Fundamentals

CUDA streams represent ordered sequences of GPU operations that execute asynchronously with respect to the host and other streams, enabling sophisticated concurrency patterns.

## Stream Types and Properties

### Stream Hierarchy and Characteristics

1. **Default Stream (Stream 0)** - Synchronous Behavior
   - Synchronous with host
   - Blocks other streams until completion
   - Used when no explicit stream specified

2. **Explicit Streams** - Asynchronous Behavior
   - Asynchronous with host
   - Can execute concurrently with other streams
   - Enable overlap and pipelining

3. **Stream Priorities**
   - Streams can be assigned priorities to hint the scheduler.
   - Ranges from high (lower number) to low (higher number).

### Stream Execution Model

1. **FIFO Ordering Within Streams**: Operations within each stream execute in submission order.
2. **Inter-Stream Concurrency**: Different streams can execute concurrently. GPU scheduler interleaves stream operations.
3. **Synchronization**:
   - `cudaStreamSynchronize()`: Wait for specific stream.
   - `cudaDeviceSynchronize()`: Wait for all streams.
   - Events: Fine-grained inter-stream dependencies.

## Stream Management Patterns

### Advanced Stream Management

For production applications, managing multiple streams efficiently is crucial. The `StreamManager` class provides a robust way to handle stream pools, priorities, and workload assignment.

**Source Code**: [`../src/04_streams_concurrency/stream_manager.cuh`](../src/04_streams_concurrency/stream_manager.cuh)

```cpp
#include "../src/04_streams_concurrency/stream_manager.cuh"

void demonstrate_stream_manager() {
    printf("=== Stream Manager Demo ===\n");

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


