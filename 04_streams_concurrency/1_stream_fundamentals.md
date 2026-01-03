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

    // Create manager with priority streams and custom tags
    std::vector<std::string> tags = {"MemoryOps", "Compute", "HighPrio", "Background"};
    StreamManager manager(4, true, tags);

    // Simulate different workload assignments
    const int N = 1024 * 1024;
    float *d_data1, *d_data2, *d_data3, *d_data4;

    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMalloc(&d_data3, N * sizeof(float));
    cudaMalloc(&d_data4, N * sizeof(float));

    // Assign workloads to appropriate streams
    cudaStream_t memory_stream = manager.get_stream_for_workload("memory_intensive");
    cudaStream_t compute_stream = manager.get_stream_for_workload("compute_intensive");
    cudaStream_t priority_stream = manager.get_stream_for_workload("high_priority");
    cudaStream_t background_stream = manager.get_next_stream();

    // ... launch operations ...

    manager.print_status();
    manager.synchronize_all();
}
```

## Nsight Debugging Tips

- Use **Nsight Systems** to visualize:
  - Stream timelines
  - Overlap of memcopy and kernels
- Identify serialization caused by:
  - Shared resources
  - Host sync calls (`cudaDeviceSynchronize()`)
