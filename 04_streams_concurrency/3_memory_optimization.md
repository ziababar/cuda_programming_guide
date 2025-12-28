#  Memory Transfer Optimization

Memory transfer optimization is crucial for achieving peak performance in CUDA applications. Understanding the memory hierarchy, transfer patterns, and bandwidth utilization strategies can significantly impact overall application throughput.

##  Pinned Memory Deep Dive

Pinned (page-locked) memory is essential for achieving maximum memory transfer bandwidth and enabling true asynchronous operations.

###  Comprehensive Pinned Memory Management
```cpp
// Advanced pinned memory allocation and management
class PinnedMemoryManager {
private:
    std::map<void*, size_t> allocated_blocks;
    std::map<void*, cudaHostAllocFlags> allocation_flags;
    size_t total_allocated;
    size_t max_allocation_limit;

public:
    PinnedMemoryManager(size_t max_limit = 2ULL * 1024 * 1024 * 1024) // 2GB default
        : total_allocated(0), max_allocation_limit(max_limit) {
        printf("PinnedMemoryManager initialized (max: %.2f GB)\n",
               max_limit / (1024.0 * 1024.0 * 1024.0));
    }

    // Allocate pinned memory with various flags
    void* allocate(size_t size, cudaHostAllocFlags flags = cudaHostAllocDefault) {
        if (total_allocated + size > max_allocation_limit) {
            printf("Warning: Allocation would exceed limit (%.2f GB used of %.2f GB)\n",
                   total_allocated / (1024.0 * 1024.0 * 1024.0),
                   max_allocation_limit / (1024.0 * 1024.0 * 1024.0));
            return nullptr;
        }

        void* ptr = nullptr;
        cudaError_t result = cudaHostAlloc(&ptr, size, flags);

        if (result == cudaSuccess && ptr != nullptr) {
            allocated_blocks[ptr] = size;
            allocation_flags[ptr] = flags;
            total_allocated += size;

            printf("Allocated %.2f MB pinned memory (flags: %d)\n",
                   size / (1024.0 * 1024.0), flags);

            return ptr;
        } else {
            printf("Failed to allocate pinned memory: %s\n", cudaGetErrorString(result));
            return nullptr;
        }
    }

    // Free pinned memory
    void deallocate(void* ptr) {
        auto it = allocated_blocks.find(ptr);
        if (it != allocated_blocks.end()) {
            size_t size = it->second;
            total_allocated -= size;

            cudaFreeHost(ptr);
            allocated_blocks.erase(it);
            allocation_flags.erase(ptr);

            printf("Freed %.2f MB pinned memory\n", size / (1024.0 * 1024.0));
        }
    }

    // Get memory statistics
    void print_statistics() {
        printf("=== Pinned Memory Statistics ===\n");
        printf("Total allocated: %.2f MB\n", total_allocated / (1024.0 * 1024.0));
        printf("Number of blocks: %zu\n", allocated_blocks.size());
        printf("Utilization: %.1f%%\n",
               (total_allocated * 100.0) / max_allocation_limit);

        // Break down by allocation flags
        std::map<cudaHostAllocFlags, size_t> flag_usage;
        for (const auto& pair : allocation_flags) {
            flag_usage[pair.second] += allocated_blocks[pair.first];
        }

        for (const auto& pair : flag_usage) {
            printf("Flag %d usage: %.2f MB\n",
                   pair.first, pair.second / (1024.0 * 1024.0));
        }
        printf("===============================\n");
    }

    ~PinnedMemoryManager() {
        // Free all remaining allocations
        for (auto& pair : allocated_blocks) {
            cudaFreeHost(pair.first);
        }
        printf("PinnedMemoryManager cleanup complete (freed %.2f MB)\n",
               total_allocated / (1024.0 * 1024.0));
    }
};

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

##  Bandwidth Optimization Strategies

###  Memory Transfer Pattern Analysis
```cpp
// Comprehensive bandwidth optimization techniques
class BandwidthOptimizer {
private:
    cudaDeviceProp device_props;
    float theoretical_bandwidth;

public:
    BandwidthOptimizer() {
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&device_props, device);

        // Calculate theoretical bandwidth
        theoretical_bandwidth = 2.0f * device_props.memoryClockRate *
                               (device_props.memoryBusWidth / 8) / 1.0e6;

        printf("=== Bandwidth Optimizer ===\n");
        printf("Device: %s\n", device_props.name);
        printf("Memory Clock Rate: %d kHz\n", device_props.memoryClockRate);
        printf("Memory Bus Width: %d bits\n", device_props.memoryBusWidth);
        printf("Theoretical Bandwidth: %.2f GB/s\n", theoretical_bandwidth);
        printf("===========================\n\n");
    }

    // Test different transfer sizes to find optimal chunk size
    void optimize_transfer_size() {
        printf("Optimizing Transfer Size:\n");

        const size_t max_size = 256 * 1024 * 1024; // 256MB
        const int num_iterations = 20;

        std::vector<size_t> test_sizes = {
            4 * 1024,           // 4KB
            64 * 1024,          // 64KB
            1024 * 1024,        // 1MB
            16 * 1024 * 1024,   // 16MB
            64 * 1024 * 1024,   // 64MB
            256 * 1024 * 1024   // 256MB
        };

        float *h_data, *d_data;
        cudaHostAlloc(&h_data, max_size, cudaHostAllocDefault);
        cudaMalloc(&d_data, max_size);

        // Initialize data
        for (size_t i = 0; i < max_size / sizeof(float); i++) {
            h_data[i] = i * 0.001f;
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        size_t optimal_size = 0;
        float best_bandwidth = 0.0f;

        for (size_t test_size : test_sizes) {
            // Measure H2D bandwidth
            cudaEventRecord(start);
            for (int i = 0; i < num_iterations; i++) {
                cudaMemcpyAsync(d_data, h_data, test_size,
                              cudaMemcpyHostToDevice, stream);
            }
            cudaStreamSynchronize(stream);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start, stop);
            float bandwidth = (test_size * num_iterations) /
                            (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

            printf("  Size: %8zu KB, Bandwidth: %6.2f GB/s (%.1f%% of theoretical)\n",
                   test_size / 1024, bandwidth,
                   (bandwidth / theoretical_bandwidth) * 100.0f);

            if (bandwidth > best_bandwidth) {
                best_bandwidth = bandwidth;
                optimal_size = test_size;
            }
        }

        printf("  Optimal transfer size: %zu KB (%.2f GB/s)\n\n",
               optimal_size / 1024, best_bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        cudaFreeHost(h_data);
        cudaFree(d_data);
    }

    // Test concurrent transfers with different stream counts
    void optimize_stream_count() {
        printf("Optimizing Stream Count for Concurrent Transfers:\n");

        const size_t transfer_size = 64 * 1024 * 1024; // 64MB per transfer
        const int max_streams = 8;

        for (int num_streams = 1; num_streams <= max_streams; num_streams++) {
            float total_bandwidth = test_concurrent_transfers(num_streams, transfer_size);
            printf("  %d streams: %.2f GB/s total bandwidth\n",
                   num_streams, total_bandwidth);
        }
        printf("\n");
    }

private:
    float test_concurrent_transfers(int num_streams, size_t transfer_size) {
        std::vector<float*> h_buffers(num_streams);
        std::vector<float*> d_buffers(num_streams);
        std::vector<cudaStream_t> streams(num_streams);

        // Allocate resources
        for (int i = 0; i < num_streams; i++) {
            cudaHostAlloc(&h_buffers[i], transfer_size, cudaHostAllocDefault);
            cudaMalloc(&d_buffers[i], transfer_size);
            cudaStreamCreate(&streams[i]);

            // Initialize data
            for (size_t j = 0; j < transfer_size / sizeof(float); j++) {
                h_buffers[i][j] = (i * 1000 + j) * 0.001f;
            }
        }

        // Measure concurrent transfers
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Launch all transfers concurrently
        for (int i = 0; i < num_streams; i++) {
            cudaMemcpyAsync(d_buffers[i], h_buffers[i], transfer_size,
                          cudaMemcpyHostToDevice, streams[i]);
        }

        // Wait for all transfers to complete
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        float total_bandwidth = (transfer_size * num_streams) /
                              (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        // Cleanup
        for (int i = 0; i < num_streams; i++) {
            cudaFreeHost(h_buffers[i]);
            cudaFree(d_buffers[i]);
            cudaStreamDestroy(streams[i]);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return total_bandwidth;
    }
};
```

##  Advanced Transfer Patterns

###  Bidirectional Transfer Optimization
```cpp
// Sophisticated bidirectional transfer patterns
class BidirectionalTransferManager {
private:
    struct TransferBuffer {
        float* h_buffer;
        float* d_buffer;
        size_t size;
        cudaStream_t upload_stream;
        cudaStream_t download_stream;
        cudaEvent_t upload_complete;
        cudaEvent_t download_complete;
    };

    std::vector<TransferBuffer> buffers;
    int num_buffers;

public:
    BidirectionalTransferManager(int buffer_count, size_t buffer_size)
        : num_buffers(buffer_count) {

        buffers.resize(buffer_count);

        for (int i = 0; i < buffer_count; i++) {
            // Allocate pinned host memory
            cudaHostAlloc(&buffers[i].h_buffer, buffer_size, cudaHostAllocDefault);

            // Allocate device memory
            cudaMalloc(&buffers[i].d_buffer, buffer_size);

            // Create dedicated streams for each direction
            cudaStreamCreate(&buffers[i].upload_stream);
            cudaStreamCreate(&buffers[i].download_stream);

            // Create events for synchronization
            cudaEventCreate(&buffers[i].upload_complete);
            cudaEventCreate(&buffers[i].download_complete);

            buffers[i].size = buffer_size;

            // Initialize with test data
            for (size_t j = 0; j < buffer_size / sizeof(float); j++) {
                buffers[i].h_buffer[j] = (i * 10000 + j) * 0.0001f;
            }
        }

        printf("BidirectionalTransferManager initialized with %d buffers\n", buffer_count);
    }

    // Demonstrate overlapped bidirectional transfers
    void demonstrate_bidirectional_overlap() {
        printf("=== Bidirectional Transfer Overlap ===\n");

        cudaEvent_t overall_start, overall_stop;
        cudaEventCreate(&overall_start);
        cudaEventCreate(&overall_stop);

        // Method 1: Sequential transfers
        printf("1. Sequential bidirectional transfers:\n");
        cudaEventRecord(overall_start);

        for (int i = 0; i < num_buffers; i++) {
            // Upload then download sequentially
            cudaMemcpy(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                      cudaMemcpyHostToDevice);
            cudaMemcpy(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                      cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(overall_stop);
        cudaEventSynchronize(overall_stop);

        float sequential_time;
        cudaEventElapsedTime(&sequential_time, overall_start, overall_stop);
        printf("   Sequential time: %.2f ms\n", sequential_time);

        // Method 2: Overlapped transfers using streams
        printf("2. Overlapped bidirectional transfers:\n");
        cudaEventRecord(overall_start);

        // Launch all uploads first
        for (int i = 0; i < num_buffers; i++) {
            cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                          cudaMemcpyHostToDevice, buffers[i].upload_stream);
            cudaEventRecord(buffers[i].upload_complete, buffers[i].upload_stream);
        }

        // Launch downloads that depend on uploads
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamWaitEvent(buffers[i].download_stream, buffers[i].upload_complete, 0);
            cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                          cudaMemcpyDeviceToHost, buffers[i].download_stream);
            cudaEventRecord(buffers[i].download_complete, buffers[i].download_stream);
        }

        // Wait for all downloads to complete
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamSynchronize(buffers[i].download_stream);
        }

        cudaEventRecord(overall_stop);
        cudaEventSynchronize(overall_stop);

        float overlapped_time;
        cudaEventElapsedTime(&overlapped_time, overall_start, overall_stop);
        printf("   Overlapped time: %.2f ms\n", overlapped_time);
        printf("   Speedup: %.2fx\n", sequential_time / overlapped_time);

        // Method 3: Advanced pipeline with computation
        printf("3. Pipeline with computation overlap:\n");
        pipeline_with_computation();

        cudaEventDestroy(overall_start);
        cudaEventDestroy(overall_stop);
    }

    void pipeline_with_computation() {
        cudaEvent_t pipeline_start, pipeline_stop;
        cudaEventCreate(&pipeline_start);
        cudaEventCreate(&pipeline_stop);

        cudaEventRecord(pipeline_start);

        // Complex pipeline: Upload -> Compute -> Download
        for (int iteration = 0; iteration < 3; iteration++) {
            for (int i = 0; i < num_buffers; i++) {
                // Stage 1: Upload data
                cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                              cudaMemcpyHostToDevice, buffers[i].upload_stream);
                cudaEventRecord(buffers[i].upload_complete, buffers[i].upload_stream);

                // Stage 2: Compute (depends on upload)
                cudaStreamWaitEvent(buffers[i].download_stream, buffers[i].upload_complete, 0);

                int num_elements = buffers[i].size / sizeof(float);
                bidirectional_compute_kernel<<<(num_elements + 255)/256, 256, 0,
                                             buffers[i].download_stream>>>(
                    buffers[i].d_buffer, num_elements, iteration);

                // Stage 3: Download result
                cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                              cudaMemcpyDeviceToHost, buffers[i].download_stream);
                cudaEventRecord(buffers[i].download_complete, buffers[i].download_stream);
            }

            // Wait for this iteration to complete before starting next
            for (int i = 0; i < num_buffers; i++) {
                cudaStreamSynchronize(buffers[i].download_stream);
            }

            printf("   Pipeline iteration %d complete\n", iteration + 1);
        }

        cudaEventRecord(pipeline_stop);
        cudaEventSynchronize(pipeline_stop);

        float pipeline_time;
        cudaEventElapsedTime(&pipeline_time, pipeline_start, pipeline_stop);
        printf("   Total pipeline time: %.2f ms\n", pipeline_time);

        cudaEventDestroy(pipeline_start);
        cudaEventDestroy(pipeline_stop);
    }

    // Analyze transfer bandwidth utilization
    void analyze_bandwidth_utilization() {
        printf("=== Bandwidth Utilization Analysis ===\n");

        const int num_measurements = 10;
        float total_bandwidth = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int measurement = 0; measurement < num_measurements; measurement++) {
            cudaEventRecord(start);

            // Concurrent bidirectional transfers
            for (int i = 0; i < num_buffers; i++) {
                cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                              cudaMemcpyHostToDevice, buffers[i].upload_stream);
                cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                              cudaMemcpyDeviceToHost, buffers[i].download_stream);
            }

            // Synchronize all transfers
            for (int i = 0; i < num_buffers; i++) {
                cudaStreamSynchronize(buffers[i].upload_stream);
                cudaStreamSynchronize(buffers[i].download_stream);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start, stop);

            // Calculate total bandwidth (upload + download)
            size_t total_bytes = buffers[0].size * num_buffers * 2; // *2 for bidirectional
            float bandwidth = total_bytes / (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            total_bandwidth += bandwidth;

            printf("   Measurement %d: %.2f GB/s\n", measurement + 1, bandwidth);
        }

        float avg_bandwidth = total_bandwidth / num_measurements;
        printf("   Average bandwidth: %.2f GB/s\n", avg_bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    ~BidirectionalTransferManager() {
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamSynchronize(buffers[i].upload_stream);
            cudaStreamSynchronize(buffers[i].download_stream);

            cudaFreeHost(buffers[i].h_buffer);
            cudaFree(buffers[i].d_buffer);
            cudaStreamDestroy(buffers[i].upload_stream);
            cudaStreamDestroy(buffers[i].download_stream);
            cudaEventDestroy(buffers[i].upload_complete);
            cudaEventDestroy(buffers[i].download_complete);
        }

        printf("BidirectionalTransferManager cleanup complete\n");
    }
};

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
