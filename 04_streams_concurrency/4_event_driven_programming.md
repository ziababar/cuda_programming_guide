# Event-Driven Programming

CUDA events provide precise synchronization control and performance measurement capabilities, enabling sophisticated coordination between streams and accurate timing analysis of GPU operations.

**[Back to Index](../README.md)** | **Previous: [Memory Transfer Optimization](3_memory_transfer.md)** | **Next: [CUDA Graphs](5_cuda_graphs.md)**

---

## **Event Fundamentals and Types**

CUDA events serve as lightweight synchronization primitives that can be recorded in streams and waited upon, providing fine-grained control over execution dependencies.

**Source Code**: [`EventManager.h`](../../src/04_streams_concurrency/EventManager.h)

```cpp
#include "../src/04_streams_concurrency/EventManager.h"

// Demonstrate different event types and their characteristics
void demonstrate_event_types() {
    printf("=== CUDA Event Types Demonstration ===\n");

    EventManager manager(16);

    // Create different types of events
    int default_event = manager.create_event("default_event", cudaEventDefault, true);
    int blocking_sync_event = manager.create_event("blocking_sync", cudaEventBlockingSync, true);
    int disable_timing_event = manager.create_event("no_timing", cudaEventDisableTiming, false);
    int interprocess_event = manager.create_event("interprocess", cudaEventInterprocess, true);

    // Create streams for testing
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Test data
    const int N = 1024 * 1024;
    float *d_data1, *d_data2;
    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));

    printf("\n1. Testing Default Event (cudaEventDefault):\n");
    printf("   - Standard event with timing capability\n");
    printf("   - Non-blocking host synchronization\n");

    manager.record_event("default_event", stream1);
    event_demo_kernel<<<(N+255)/256, 256, 0, stream1>>>(d_data1, N, 1);
    manager.record_event("default_event", stream1);

    // Check completion status
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    bool complete = manager.is_event_complete("default_event");
    printf("   Event complete after 10ms: %s\n", complete ? "yes" : "no");

    manager.synchronize_event("default_event");
    printf("   Event synchronized successfully\n");

    printf("\n2. Testing Blocking Sync Event (cudaEventBlockingSync):\n");
    printf("   - Uses blocking synchronization (lower CPU usage)\n");
    printf("   - More efficient for host threads that wait\n");

    manager.record_event("blocking_sync", stream2);
    event_demo_kernel<<<(N+255)/256, 256, 0, stream2>>>(d_data2, N, 2);
    manager.record_event("blocking_sync", stream2);

    auto sync_start = std::chrono::high_resolution_clock::now();
    manager.synchronize_event("blocking_sync");
    auto sync_end = std::chrono::high_resolution_clock::now();

    auto sync_time = std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_start);
    printf("   Blocking sync time: %ld μs\n", sync_time.count());

    printf("\n3. Testing Timing-Disabled Event (cudaEventDisableTiming):\n");
    printf("   - Lower overhead, no timing capability\n");
    printf("   - Optimized for synchronization-only use cases\n");

    manager.record_event("no_timing", stream1);
    event_demo_kernel<<<(N+255)/256, 256, 0, stream1>>>(d_data1, N, 3);

    // Try to measure time (should fail gracefully)
    float invalid_time = manager.get_elapsed_time("no_timing", "default_event");
    printf("   Timing measurement result: %.3f ms (expected: -1.0)\n", invalid_time);

    printf("\n4. Inter-Stream Dependencies:\n");
    printf("   Using events to coordinate between streams\n");

    // Stream 1 does initial work
    manager.record_event("stream1_complete", stream1);
    event_demo_kernel<<<(N+255)/256, 256, 0, stream1>>>(d_data1, N, 4);
    manager.record_event("stream1_complete", stream1);

    // Stream 2 waits for Stream 1 to complete
    manager.stream_wait_event(stream2, "stream1_complete");
    event_demo_kernel<<<(N+255)/256, 256, 0, stream2>>>(d_data2, N, 5);

    // Synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    printf("   Inter-stream dependency executed successfully\n");

    // Print final statistics
    manager.print_event_statistics();

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data1);
    cudaFree(d_data2);
}

__global__ void event_demo_kernel(float* data, int N, int kernel_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = tid * 0.001f;

        // Different computation patterns per kernel
        for (int i = 0; i < kernel_id * 20; i++) {
            value = sin(value) + cos(value * 0.1f);
        }

        data[tid] = value;
    }

    // First thread reports completion
    if (tid == 0) {
        printf("Kernel %d execution complete\n", kernel_id);
    }
}
```

## **Precision Timing and Performance Measurement**

Events provide the most accurate method for measuring GPU execution times, with sub-millisecond precision and minimal overhead.

**Source Code**: [`PerformanceProfiler.h`](../../src/04_streams_concurrency/PerformanceProfiler.h)

```cpp
#include "../src/04_streams_concurrency/PerformanceProfiler.h"

// Comprehensive timing demonstration
void demonstrate_performance_profiling() {
    printf("=== Performance Profiling Demonstration ===\n");

    PerformanceProfiler profiler;

    // Create test data
    const int N = 2 * 1024 * 1024; // 2M elements
    float *h_data, *d_input, *d_output, *d_temp;

    cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = sin(i * 0.001f);
    }

    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    printf("\n1. Basic Operation Timing:\n");

    // Time memory transfer
    profiler.start_timing("memory_h2d");
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    profiler.stop_timing("memory_h2d");

    // Time kernel execution
    profiler.start_timing("kernel_execution");
    complex_math_kernel<<<(N+255)/256, 256>>>(d_input, d_output, N);
    profiler.stop_timing("kernel_execution");

    // Time memory transfer back
    profiler.start_timing("memory_d2h");
    cudaMemcpy(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    profiler.stop_timing("memory_d2h");

    printf("\n2. Scope-Based Timing (RAII):\n");
    {
        auto timer = profiler.time_scope("scoped_operation");

        // Complex operation
        cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        complex_math_kernel<<<(N+255)/256, 256>>>(d_input, d_temp, N);
        simple_math_kernel<<<(N+255)/256, 256>>>(d_temp, d_output, N);
        cudaMemcpy(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Timer automatically stops when scope ends
    }

    printf("\n3. Benchmarking Operations:\n");

    // Benchmark different kernel configurations
    profiler.benchmark_operation("kernel_256_threads", [&]() {
        complex_math_kernel<<<(N+255)/256, 256>>>(d_input, d_output, N);
        cudaDeviceSynchronize();
    }, 5);

    profiler.benchmark_operation("kernel_512_threads", [&]() {
        complex_math_kernel<<<(N+511)/512, 512>>>(d_input, d_output, N);
        cudaDeviceSynchronize();
    }, 5);

    profiler.benchmark_operation("kernel_1024_threads", [&]() {
        complex_math_kernel<<<(N+1023)/1024, 1024>>>(d_input, d_output, N);
        cudaDeviceSynchronize();
    }, 5);

    printf("\n4. Pipeline Timing:\n");

    // Time overlapped operations
    profiler.start_timing("overlapped_pipeline");

    cudaMemcpyAsync(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream1);
    complex_math_kernel<<<(N+255)/256, 256, 0, stream1>>>(d_input, d_temp, N);
    simple_math_kernel<<<(N+255)/256, 256, 0, stream2>>>(d_temp, d_output, N);
    cudaMemcpyAsync(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    profiler.stop_timing("overlapped_pipeline");

    // Print comprehensive results
    profiler.print_all_stats();

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

__global__ void complex_math_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = input[tid];

        // Complex mathematical operations
        for (int i = 0; i < 50; i++) {
            value = sin(value) + cos(value);
            value = sqrt(fabs(value) + 1.0f);
            value = log(value + 1.0f);
        }

        output[tid] = value;
    }
}

__global__ void simple_math_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        output[tid] = input[tid] * 2.0f + 1.0f;
    }
}
```

## **Advanced Synchronization Patterns**

Events enable sophisticated synchronization patterns beyond basic stream coordination, including complex dependency graphs and multi-stage pipeline coordination.

**Source Code**: [`EventCoordinator.h`](../../src/04_streams_concurrency/EventCoordinator.h)

```cpp
#include "../src/04_streams_concurrency/EventCoordinator.h"
```
