#ifndef STREAM_MANAGER_CUH
#define STREAM_MANAGER_CUH

#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <iostream>

// Forward declarations
inline __global__ void preprocessing_kernel(float* data, int N);
inline __global__ void processing_kernel(float* data, int N);
inline __global__ void compute_intensive_kernel(float* data, int N);

// Sophisticated stream management for production applications
class StreamManager {
private:
    std::vector<cudaStream_t> streams;
    std::vector<int> stream_priorities;
    std::vector<bool> stream_busy;
    std::vector<std::string> stream_tags;
    int num_streams;
    int current_stream_index;

public:
    StreamManager(int count, bool use_priorities = false,
                 const std::vector<std::string>& tags = {})
        : num_streams(count), current_stream_index(0) {

        streams.resize(count);
        stream_priorities.resize(count);
        stream_busy.resize(count, false);
        stream_tags.resize(count);

        // Set up stream tags
        for (int i = 0; i < count; i++) {
            if (i < tags.size()) {
                stream_tags[i] = tags[i];
            } else {
                stream_tags[i] = "Stream_" + std::to_string(i);
            }
        }

        if (use_priorities) {
            int low_priority, high_priority;
            cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);

            // Distribute priorities evenly across streams
            for (int i = 0; i < count; i++) {
                int priority = high_priority +
                              (i * (low_priority - high_priority)) / std::max(1, count - 1);
                stream_priorities[i] = priority;

                cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, priority);
                printf("Created %s with priority %d\n", stream_tags[i].c_str(), priority);
            }
        } else {
            // Standard streams with no explicit priorities
            for (int i = 0; i < count; i++) {
                cudaStreamCreate(&streams[i]);
                stream_priorities[i] = 0;
                printf("Created %s (standard priority)\n", stream_tags[i].c_str());
            }
        }

        printf("StreamManager initialized with %d streams\n", count);
    }

    // Get stream by index
    cudaStream_t get_stream(int index) const {
        if (index >= 0 && index < num_streams) {
            return streams[index];
        }
        return streams[0];  // Fallback to first stream
    }

    // Round-robin stream allocation
    cudaStream_t get_next_stream() {
        cudaStream_t stream = streams[current_stream_index];
        current_stream_index = (current_stream_index + 1) % num_streams;
        return stream;
    }

    // Get least busy stream
    cudaStream_t get_available_stream() {
        // First, try to find a completely idle stream
        for (int i = 0; i < num_streams; i++) {
            cudaError_t status = cudaStreamQuery(streams[i]);
            if (status == cudaSuccess) {  // Stream is idle
                stream_busy[i] = false;
                return streams[i];
            }
        }

        // If all streams are busy, return the next one in round-robin
        return get_next_stream();
    }

    // Get stream optimized for specific workload
    cudaStream_t get_stream_for_workload(const std::string& workload_type) {
        if (workload_type == "high_priority") {
            // Return highest priority stream
            int best_stream = 0;
            int best_priority = stream_priorities[0];

            for (int i = 1; i < num_streams; i++) {
                if (stream_priorities[i] < best_priority) {  // Lower number = higher priority
                    best_priority = stream_priorities[i];
                    best_stream = i;
                }
            }

            printf("Assigned %s for high priority workload\n", stream_tags[best_stream].c_str());
            return streams[best_stream];

        } else if (workload_type == "memory_intensive") {
            // For memory-intensive work, prefer available streams to avoid competition
            return get_available_stream();

        } else if (workload_type == "compute_intensive") {
            // For compute-intensive work, any stream is fine
            return get_next_stream();

        } else {
            // Default case
            return get_available_stream();
        }
    }

    // Synchronize specific stream
    void synchronize_stream(int index) {
        if (index >= 0 && index < num_streams) {
            cudaStreamSynchronize(streams[index]);
            stream_busy[index] = false;
            printf("%s synchronized\n", stream_tags[index].c_str());
        }
    }

    // Synchronize all streams
    void synchronize_all() {
        printf("Synchronizing all streams...\n");
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
            stream_busy[i] = false;
        }
        printf("All streams synchronized\n");
    }

    // Query status of all streams
    void print_status() {
        printf("=== Stream Status ===\n");
        for (int i = 0; i < num_streams; i++) {
            cudaError_t status = cudaStreamQuery(streams[i]);
            const char* status_str = (status == cudaSuccess) ? "Idle" : "Busy";
            printf("  %s (priority %d): %s\n",
                   stream_tags[i].c_str(), stream_priorities[i], status_str);
        }
        printf("====================\n");
    }

    // Get performance statistics
    void print_performance_stats() {
        printf("=== Stream Performance Stats ===\n");
        // This would require more sophisticated tracking in a real implementation
        printf("Total streams: %d\n", num_streams);
        printf("Priority range: %d to %d\n",
               *std::min_element(stream_priorities.begin(), stream_priorities.end()),
               *std::max_element(stream_priorities.begin(), stream_priorities.end()));
        printf("Current allocation index: %d\n", current_stream_index);
        printf("===============================\n");
    }

    ~StreamManager() {
        printf("Destroying StreamManager...\n");
        synchronize_all();

        for (int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(streams[i]);
            printf("Destroyed %s\n", stream_tags[i].c_str());
        }

        printf("StreamManager cleanup complete\n");
    }
};

// Usage example
inline void demonstrate_stream_manager() {
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

#endif // STREAM_MANAGER_CUH
