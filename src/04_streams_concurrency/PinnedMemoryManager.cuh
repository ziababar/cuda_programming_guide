#ifndef PINNED_MEMORY_MANAGER_CUH
#define PINNED_MEMORY_MANAGER_CUH

#include <map>
#include <cstdio>
#include <vector>
#include <cmath>

// Forward declaration
inline __global__ void zero_copy_kernel(float* data, int N);

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
inline void demonstrate_pinned_memory_types() {
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

inline __global__ void zero_copy_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Direct access to host memory (zero-copy)
        data[tid] = sqrt(data[tid]) + 1.0f;
    }
}

#endif // PINNED_MEMORY_MANAGER_CUH
