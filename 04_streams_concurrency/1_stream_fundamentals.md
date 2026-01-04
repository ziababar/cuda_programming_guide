# Stream Fundamentals

CUDA streams represent ordered sequences of GPU operations that execute asynchronously with respect to the host and other streams, enabling sophisticated concurrency patterns.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Related: [Memory Hierarchy](../02_memory_hierarchy/1_cuda_memory_hierarchy.md)**

---

## Stream Types and Properties

### Stream Hierarchy and Characteristics

```cpp
// Comprehensive stream type demonstration
void demonstrate_stream_fundamentals() {
    printf("=== CUDA Stream Fundamentals ===\n");

    // 1. Default Stream (Stream 0) - Synchronous Behavior
    printf("1. Default Stream Characteristics:\n");
    printf("   - Synchronous with host\n");
    printf("   - Blocks other streams until completion\n");
    printf("   - Used when no explicit stream specified\n\n");

    float *d_data1, *d_data2;
    size_t size = 1024 * sizeof(float);

    cudaMalloc(&d_data1, size);
    cudaMalloc(&d_data2, size);

    // Default stream operations execute sequentially
    auto start = std::chrono::high_resolution_clock::now();

    cudaMemset(d_data1, 0, size);                    // Blocks host
    simple_kernel<<<256, 256>>>(d_data1, 1024);     // Blocks until memset done
    cudaMemset(d_data2, 1, size);                    // Blocks until kernel done
    cudaDeviceSynchronize();                         // Wait for completion

    auto end = std::chrono::high_resolution_clock::now();
    auto sequential_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("   Sequential execution time: %ld μs\n\n", sequential_time.count());

    // 2. Explicit Streams - Asynchronous Behavior
    printf("2. Explicit Stream Characteristics:\n");
    printf("   - Asynchronous with host\n");
    printf("   - Can execute concurrently with other streams\n");
    printf("   - Enable overlap and pipelining\n");

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    start = std::chrono::high_resolution_clock::now();

    // These can execute concurrently
    cudaMemsetAsync(d_data1, 0, size, stream1);
    cudaMemsetAsync(d_data2, 1, size, stream2);
    simple_kernel<<<256, 256, 0, stream1>>>(d_data1, 1024);
    simple_kernel<<<256, 256, 0, stream2>>>(d_data2, 1024);

    // Synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    end = std::chrono::high_resolution_clock::now();
    auto concurrent_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("   Concurrent execution time: %ld μs\n", concurrent_time.count());
    printf("   Speedup: %.2fx\n\n", (float)sequential_time.count() / concurrent_time.count());

    // 3. Stream Properties and Configuration
    printf("3. Stream Properties:\n");

    // Query stream priorities
    int low_priority, high_priority;
    cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);
    printf("   Priority range: %d (low) to %d (high)\n", low_priority, high_priority);

    // Create priority streams
    cudaStream_t high_prio_stream, low_prio_stream;
    cudaStreamCreateWithPriority(&high_prio_stream, cudaStreamNonBlocking, high_priority);
    cudaStreamCreateWithPriority(&low_prio_stream, cudaStreamNonBlocking, low_priority);

    printf("   High priority stream created\n");
    printf("   Low priority stream created\n");

    // Test non-blocking vs blocking behavior
    cudaStream_t blocking_stream, non_blocking_stream;
    cudaStreamCreateWithFlags(&blocking_stream, cudaStreamDefault);        // Blocking
    cudaStreamCreateWithFlags(&non_blocking_stream, cudaStreamNonBlocking); // Non-blocking

    printf("   Blocking stream: synchronizes with default stream\n");
    printf("   Non-blocking stream: independent execution\n\n");

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(high_prio_stream);
    cudaStreamDestroy(low_prio_stream);
    cudaStreamDestroy(blocking_stream);
    cudaStreamDestroy(non_blocking_stream);
    cudaFree(d_data1);
    cudaFree(d_data2);
}

__global__ void simple_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Simple computation to demonstrate stream behavior
        data[tid] = tid * 2.0f + sin(tid * 0.01f);
    }
}
```

### Stream Execution Model

```cpp
// Demonstrate FIFO ordering and inter-stream concurrency
void demonstrate_stream_execution_model() {
    printf("=== Stream Execution Model ===\n");

    const int N = 1024 * 1024;
    float *h_data, *d_data1, *d_data2, *d_data3;

    // Allocate pinned memory for optimal transfer performance
    cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMalloc(&d_data3, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.001f;
    }

    // Create streams with different characteristics
    cudaStream_t stream_a, stream_b, stream_c;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);
    cudaStreamCreate(&stream_c);

    printf("1. FIFO Ordering Within Streams:\n");
    printf("   Operations within each stream execute in submission order\n");

    // Stream A: Sequential pipeline
    cudaMemcpyAsync(d_data1, h_data, N * sizeof(float),
                   cudaMemcpyHostToDevice, stream_a);         // Order: 1
    preprocessing_kernel<<<(N+255)/256, 256, 0, stream_a>>>(d_data1, N);    // Order: 2
    processing_kernel<<<(N+255)/256, 256, 0, stream_a>>>(d_data1, N);       // Order: 3
    cudaMemcpyAsync(h_data, d_data1, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_a);         // Order: 4

    printf("   Stream A: H2D -> Preprocess -> Process -> D2H\n");

    // Stream B: Different pipeline
    cudaMemcpyAsync(d_data2, h_data, N * sizeof(float),
                   cudaMemcpyHostToDevice, stream_b);         // Concurrent with Stream A
    compute_intensive_kernel<<<(N+255)/256, 256, 0, stream_b>>>(d_data2, N); // Different work
    postprocessing_kernel<<<(N+255)/256, 256, 0, stream_b>>>(d_data2, N);

    printf("   Stream B: H2D -> Intensive -> Postprocess (concurrent)\n");

    // Stream C: Memory operations
    cudaMemsetAsync(d_data3, 0, N * sizeof(float), stream_c);              // Concurrent init
    initialization_kernel<<<(N+255)/256, 256, 0, stream_c>>>(d_data3, N);

    printf("   Stream C: Memset -> Initialize (concurrent)\n\n");

    printf("2. Inter-Stream Concurrency:\n");
    printf("   Different streams can execute concurrently\n");
    printf("   GPU scheduler interleaves stream operations\n");
    printf("   Actual concurrency depends on resource availability\n\n");

    // Wait for all streams to complete
    cudaStreamSynchronize(stream_a);
    cudaStreamSynchronize(stream_b);
    cudaStreamSynchronize(stream_c);

    printf("3. Synchronization Points:\n");
    printf("   cudaStreamSynchronize() - Wait for specific stream\n");
    printf("   cudaDeviceSynchronize() - Wait for all streams\n");
    printf("   Events - Fine-grained inter-stream dependencies\n");

    // Cleanup
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    cudaStreamDestroy(stream_c);
    cudaFreeHost(h_data);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
}

__global__ void preprocessing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = sqrt(fabs(data[tid]));
    }
}

__global__ void processing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = sin(data[tid]) + cos(data[tid]);
    }
}

__global__ void compute_intensive_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float value = data[tid];
        // Intensive computation
        for (int i = 0; i < 100; i++) {
            value = sin(value) + cos(value);
        }
        data[tid] = value;
    }
}

__global__ void postprocessing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

__global__ void initialization_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = tid * 0.01f;
    }
}
```

## Stream Management Patterns

### Advanced Stream Management

For sophisticated stream management, we can use a `StreamManager` class.

**Source Code**: [`../src/04_streams_concurrency/stream_manager.h`](../src/04_streams_concurrency/stream_manager.h)

```cpp
// Usage example
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

    // Launch operations
    cudaMemsetAsync(d_data1, 0, N * sizeof(float), memory_stream);
    compute_intensive_kernel<<<(N+255)/256, 256, 0, compute_stream>>>(d_data2, N);
    preprocessing_kernel<<<(N+255)/256, 256, 0, priority_stream>>>(d_data3, N);
    processing_kernel<<<(N+255)/256, 256, 0, background_stream>>>(d_data4, N);

    // Check status
    manager.print_status();

    // Synchronize and get final status
    manager.synchronize_all();
    manager.print_performance_stats();

    // Cleanup
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
    cudaFree(d_data4);
}
```

## Debugging Streams

When working with streams, it is crucial to understand synchronization points and potential serialization.

### Nsight Debugging Tips

- Use **Nsight Systems** to visualize:
  - Stream timelines
  - Overlap of memcopy and kernels
- Identify serialization caused by:
  - Shared resources
  - Host sync calls (`cudaDeviceSynchronize()`)
