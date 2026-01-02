#ifndef ADAPTIVE_STREAM_MANAGER_CUH
#define ADAPTIVE_STREAM_MANAGER_CUH

#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <map>
#include <chrono>
#include <cstdio>
#include <algorithm>

// Dynamic stream management for varying workloads
class AdaptiveStreamManager {
private:
    std::vector<cudaStream_t> stream_pool;
    std::queue<int> available_streams;
    std::map<int, bool> stream_busy;
    std::map<int, std::chrono::high_resolution_clock::time_point> stream_start_times;
    std::map<int, float> stream_utilization;
    int max_streams;
    float target_utilization;

public:
    AdaptiveStreamManager(int initial_streams = 4, float target_util = 0.8f)
        : max_streams(16), target_utilization(target_util) {

        // Initialize stream pool
        stream_pool.resize(max_streams);
        for (int i = 0; i < max_streams; i++) {
            stream_pool[i] = nullptr;
            stream_busy[i] = false;
            stream_utilization[i] = 0.0f;
        }

        // Create initial streams
        for (int i = 0; i < initial_streams; i++) {
            create_stream(i);
            available_streams.push(i);
        }

        printf("AdaptiveStreamManager initialized with %d streams (max: %d)\n",
               initial_streams, max_streams);
    }

    void create_stream(int index) {
        if (index < max_streams && stream_pool[index] == nullptr) {
            cudaStreamCreate(&stream_pool[index]);
            printf("Created stream %d\n", index);
        }
    }

    void destroy_stream(int index) {
        if (index < max_streams && stream_pool[index] != nullptr) {
            cudaStreamSynchronize(stream_pool[index]);
            cudaStreamDestroy(stream_pool[index]);
            stream_pool[index] = nullptr;
            stream_busy[index] = false;
            printf("Destroyed stream %d\n", index);
        }
    }

    // Get an available stream, creating new ones if needed
    int acquire_stream() {
        // First, check for available streams
        if (!available_streams.empty()) {
            int stream_id = available_streams.front();
            available_streams.pop();
            stream_busy[stream_id] = true;
            stream_start_times[stream_id] = std::chrono::high_resolution_clock::now();
            return stream_id;
        }

        // No available streams, try to create a new one
        for (int i = 0; i < max_streams; i++) {
            if (stream_pool[i] == nullptr) {
                create_stream(i);
                stream_busy[i] = true;
                stream_start_times[i] = std::chrono::high_resolution_clock::now();
                return i;
            }
        }

        // All streams are created and busy, wait for one to become available
        printf("All streams busy, waiting for availability...\n");

        // Check which streams have completed
        for (int i = 0; i < max_streams; i++) {
            if (stream_pool[i] != nullptr && stream_busy[i]) {
                cudaError_t status = cudaStreamQuery(stream_pool[i]);
                if (status == cudaSuccess) {
                    // Stream is now available
                    return acquire_completed_stream(i);
                }
            }
        }

        // Fallback: return first stream (will cause serialization)
        printf("Warning: Returning busy stream, may cause serialization\n");
        return 0;
    }

    int acquire_completed_stream(int stream_id) {
        // Update utilization statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - stream_start_times[stream_id]);

        // Simple utilization tracking (would be more sophisticated in production)
        stream_utilization[stream_id] =
            (stream_utilization[stream_id] * 0.9f) + (duration.count() * 0.1f);

        stream_busy[stream_id] = true;
        stream_start_times[stream_id] = std::chrono::high_resolution_clock::now();

        return stream_id;
    }

    void release_stream(int stream_id) {
        if (stream_id >= 0 && stream_id < max_streams && stream_busy[stream_id]) {
            stream_busy[stream_id] = false;
            available_streams.push(stream_id);

            // Update utilization
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - stream_start_times[stream_id]);

            stream_utilization[stream_id] =
                (stream_utilization[stream_id] * 0.9f) + (duration.count() * 0.1f);
        }
    }

    cudaStream_t get_cuda_stream(int stream_id) {
        if (stream_id >= 0 && stream_id < max_streams && stream_pool[stream_id] != nullptr) {
            return stream_pool[stream_id];
        }
        return nullptr;
    }

    // Adaptive management: create/destroy streams based on utilization
    void optimize_stream_count() {
        int active_streams = 0;
        float avg_utilization = 0.0f;

        for (int i = 0; i < max_streams; i++) {
            if (stream_pool[i] != nullptr) {
                active_streams++;
                avg_utilization += stream_utilization[i];
            }
        }

        if (active_streams > 0) {
            avg_utilization /= active_streams;
        }

        printf("Active streams: %d, Average utilization: %.2f\n",
               active_streams, avg_utilization);

        // Scale up if high utilization
        if (avg_utilization > target_utilization && active_streams < max_streams) {
            for (int i = 0; i < max_streams; i++) {
                if (stream_pool[i] == nullptr) {
                    create_stream(i);
                    available_streams.push(i);
                    printf("Scaling up: created stream %d\n", i);
                    break;
                }
            }
        }

        // Scale down if low utilization
        if (avg_utilization < target_utilization * 0.5f && active_streams > 2) {
            for (int i = max_streams - 1; i >= 0; i--) {
                if (stream_pool[i] != nullptr && !stream_busy[i] &&
                    stream_utilization[i] < target_utilization * 0.3f) {
                    destroy_stream(i);
                    printf("Scaling down: destroyed stream %d\n", i);
                    break;
                }
            }
        }
    }

    void print_status() {
        printf("=== Adaptive Stream Manager Status ===\n");
        printf("Available streams in queue: %zu\n", available_streams.size());

        for (int i = 0; i < max_streams; i++) {
            if (stream_pool[i] != nullptr) {
                const char* status = stream_busy[i] ? "Busy" : "Available";
                printf("Stream %d: %s (utilization: %.2f)\n",
                       i, status, stream_utilization[i]);
            }
        }
        printf("=====================================\n");
    }

    ~AdaptiveStreamManager() {
        for (int i = 0; i < max_streams; i++) {
            if (stream_pool[i] != nullptr) {
                destroy_stream(i);
            }
        }
        printf("AdaptiveStreamManager destroyed\n");
    }
};

#endif // ADAPTIVE_STREAM_MANAGER_CUH
