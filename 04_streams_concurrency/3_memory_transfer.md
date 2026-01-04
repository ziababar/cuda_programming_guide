# Memory Transfer Optimization

Memory transfer optimization is crucial for achieving peak performance in CUDA applications. Understanding the memory hierarchy, transfer patterns, and bandwidth utilization strategies can significantly impact overall application throughput.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Related: [Stream Fundamentals](1_stream_fundamentals.md)**

---

## Pinned Memory Deep Dive

Pinned (page-locked) memory is essential for achieving maximum memory transfer bandwidth and enabling true asynchronous operations.

### Pinned Memory Management

**Source Code**: [`../src/04_streams_concurrency/pinned_memory_manager.h`](../src/04_streams_concurrency/pinned_memory_manager.h)

```cpp
// Demonstrate different pinned memory allocation types
void demonstrate_pinned_memory_types() {
    printf("=== Pinned Memory Types Comparison ===\n");

    const size_t test_size = 64 * 1024 * 1024; // 64MB
    PinnedMemoryManager manager;

    // Test different allocation flags
    struct TestConfig {
        cudaHostAllocFlags flags;
        const char* description;
    } configs[] = {
        {cudaHostAllocDefault, "Default pinned memory"},
        {cudaHostAllocWriteCombined, "Write-combined (faster H2D)"},
        {cudaHostAllocMapped, "Mapped (zero-copy access)"},
        {cudaHostAllocPortable, "Portable across contexts"},
        {cudaHostAllocWriteCombined | cudaHostAllocMapped, "Write-combined + Mapped"}
    };

    std::vector<void*> test_buffers;
    float *d_buffer;
    cudaMalloc(&d_buffer, test_size);

    // Create streams for async operations
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Test each configuration
    for (const auto& config : configs) {
        printf("\nTesting: %s\n", config.description);

        void* h_buffer = manager.allocate(test_size, config.flags);
        if (!h_buffer) continue;

        test_buffers.push_back(h_buffer);

        // Initialize data
        for (int i = 0; i < test_size / sizeof(float); i++) {
            ((float*)h_buffer)[i] = i * 0.001f;
        }

        // Measure transfer performance
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Host to Device transfer
        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++) {
            cudaMemcpyAsync(d_buffer, h_buffer, test_size,
                          cudaMemcpyHostToDevice, stream1);
        }
        cudaStreamSynchronize(stream1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float h2d_time;
        cudaEventElapsedTime(&h2d_time, start, stop);
        float h2d_bandwidth = (test_size * 10) / (h2d_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        // Device to Host transfer
        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++) {
            cudaMemcpyAsync(h_buffer, d_buffer, test_size,
                          cudaMemcpyDeviceToHost, stream2);
        }
        cudaStreamSynchronize(stream2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float d2h_time;
        cudaEventElapsedTime(&d2h_time, start, stop);
        float d2h_bandwidth = (test_size * 10) / (d2h_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        printf("  H2D Bandwidth: %.2f GB/s\n", h2d_bandwidth);
        printf("  D2H Bandwidth: %.2f GB/s\n", d2h_bandwidth);

        // Test zero-copy access if mapped
        if (config.flags & cudaHostAllocMapped) {
            float* d_mapped_ptr;
            cudaHostGetDevicePointer(&d_mapped_ptr, h_buffer, 0);

            printf("  Zero-copy access enabled (device ptr: %p)\n", d_mapped_ptr);

            // Launch kernel that directly accesses host memory
            zero_copy_kernel<<<(test_size/sizeof(float) + 255)/256, 256>>>(
                d_mapped_ptr, test_size/sizeof(float));
            cudaDeviceSynchronize();

            printf("  Zero-copy kernel execution successful\n");
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Cleanup
    for (void* buffer : test_buffers) {
        manager.deallocate(buffer);
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_buffer);

    manager.print_statistics();
}

__global__ void zero_copy_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Direct access to host memory (zero-copy)
        data[tid] = sqrt(data[tid]) + 1.0f;
    }
}
```

## Bandwidth Optimization Strategies

### Memory Transfer Pattern Analysis

```cpp
// Comprehensive bandwidth optimization techniques
// Full implementation available in ../src/04_streams_concurrency/bandwidth_optimizer.cuh
#include "../src/04_streams_concurrency/bandwidth_optimizer.cuh"
```

## Advanced Transfer Patterns

### Bidirectional Transfer Optimization

```cpp
// Full implementation available in ../src/04_streams_concurrency/bidirectional_transfer_manager.cuh
#include "../src/04_streams_concurrency/bidirectional_transfer_manager.cuh"

__global__ void bidirectional_compute_kernel(float* data, int N, int iteration) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Different computation per iteration
        for (int i = 0; i < (iteration + 1) * 50; i++) {
            value = sin(value) + cos(value * 0.1f);
        }

        data[tid] = value;
    }
}
```
