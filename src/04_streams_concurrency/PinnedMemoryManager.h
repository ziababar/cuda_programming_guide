#pragma once
#include <cuda_runtime.h>
#include <map>
#include <cstdio>
#include <vector>

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
