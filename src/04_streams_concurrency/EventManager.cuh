#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <chrono>
#include <cstdio>
#include <algorithm>

__global__ void event_demo_kernel(float* data, int N, int kernel_id);

// Advanced event management system for complex applications
class EventManager {
private:
    struct EventInfo {
        cudaEvent_t event;
        std::string name;
        cudaEventFlags flags;
        bool is_timing_event;
        std::chrono::high_resolution_clock::time_point created_time;
        int usage_count;
    };

    std::vector<EventInfo> events;
    std::map<std::string, int> event_name_map;
    std::queue<int> available_events;
    int max_events;

public:
    EventManager(int max_event_count = 64) : max_events(max_event_count) {
        events.resize(max_events);

        for (int i = 0; i < max_events; i++) {
            events[i].event = nullptr;
            events[i].is_timing_event = false;
            events[i].usage_count = 0;
        }

        printf("EventManager initialized (max events: %d)\n", max_events);
    }

    // Create event with specific properties
    int create_event(const std::string& name, cudaEventFlags flags = cudaEventDefault,
                    bool for_timing = true) {

        // Find available slot
        int event_id = -1;
        for (int i = 0; i < max_events; i++) {
            if (events[i].event == nullptr) {
                event_id = i;
                break;
            }
        }

        if (event_id == -1) {
            printf("Error: No available event slots\n");
            return -1;
        }

        // Create CUDA event
        cudaError_t result = cudaEventCreateWithFlags(&events[event_id].event, flags);
        if (result != cudaSuccess) {
            printf("Failed to create event '%s': %s\n", name.c_str(), cudaGetErrorString(result));
            return -1;
        }

        // Set event properties
        events[event_id].name = name;
        events[event_id].flags = flags;
        events[event_id].is_timing_event = for_timing && !(flags & cudaEventDisableTiming);
        events[event_id].created_time = std::chrono::high_resolution_clock::now();
        events[event_id].usage_count = 0;

        // Register name mapping
        event_name_map[name] = event_id;

        printf("Created event '%s' (ID: %d, flags: %d, timing: %s)\n",
               name.c_str(), event_id, flags,
               events[event_id].is_timing_event ? "enabled" : "disabled");

        return event_id;
    }

    // Get event by ID or name
    cudaEvent_t get_event(int event_id) {
        if (event_id >= 0 && event_id < max_events && events[event_id].event != nullptr) {
            events[event_id].usage_count++;
            return events[event_id].event;
        }
        return nullptr;
    }

    cudaEvent_t get_event(const std::string& name) {
        auto it = event_name_map.find(name);
        if (it != event_name_map.end()) {
            return get_event(it->second);
        }
        return nullptr;
    }

    // Record event in stream
    void record_event(const std::string& name, cudaStream_t stream = 0) {
        cudaEvent_t event = get_event(name);
        if (event != nullptr) {
            cudaEventRecord(event, stream);
            printf("Recorded event '%s' in stream %p\n", name.c_str(), stream);
        }
    }

    void record_event(int event_id, cudaStream_t stream = 0) {
        if (event_id >= 0 && event_id < max_events && events[event_id].event != nullptr) {
            cudaEventRecord(events[event_id].event, stream);
            printf("Recorded event '%s' (ID: %d) in stream %p\n",
                   events[event_id].name.c_str(), event_id, stream);
        }
    }

    // Synchronize on event
    void synchronize_event(const std::string& name) {
        cudaEvent_t event = get_event(name);
        if (event != nullptr) {
            cudaEventSynchronize(event);
            printf("Synchronized on event '%s'\n", name.c_str());
        }
    }

    // Wait for event in stream
    void stream_wait_event(cudaStream_t stream, const std::string& event_name) {
        cudaEvent_t event = get_event(event_name);
        if (event != nullptr) {
            cudaStreamWaitEvent(stream, event, 0);
            printf("Stream %p waiting for event '%s'\n", stream, event_name.c_str());
        }
    }

    // Query event status
    bool is_event_complete(const std::string& name) {
        cudaEvent_t event = get_event(name);
        if (event != nullptr) {
            cudaError_t status = cudaEventQuery(event);
            return (status == cudaSuccess);
        }
        return false;
    }

    // Measure elapsed time between events
    float get_elapsed_time(const std::string& start_event, const std::string& stop_event) {
        cudaEvent_t start = get_event(start_event);
        cudaEvent_t stop = get_event(stop_event);

        if (start != nullptr && stop != nullptr) {
            // Check if both events are timing-enabled
            int start_id = event_name_map[start_event];
            int stop_id = event_name_map[stop_event];

            if (!events[start_id].is_timing_event || !events[stop_id].is_timing_event) {
                printf("Warning: One or both events have timing disabled\n");
                return -1.0f;
            }

            float elapsed_time;
            cudaError_t result = cudaEventElapsedTime(&elapsed_time, start, stop);

            if (result == cudaSuccess) {
                printf("Elapsed time between '%s' and '%s': %.3f ms\n",
                       start_event.c_str(), stop_event.c_str(), elapsed_time);
                return elapsed_time;
            } else {
                printf("Failed to measure elapsed time: %s\n", cudaGetErrorString(result));
            }
        }

        return -1.0f;
    }

    // Print event statistics
    void print_event_statistics() {
        printf("=== Event Manager Statistics ===\n");

        int active_events = 0;
        int timing_events = 0;
        int total_usage = 0;

        for (int i = 0; i < max_events; i++) {
            if (events[i].event != nullptr) {
                active_events++;
                total_usage += events[i].usage_count;

                if (events[i].is_timing_event) {
                    timing_events++;
                }

                printf("  Event '%s' (ID: %d): used %d times, timing: %s\n",
                       events[i].name.c_str(), i, events[i].usage_count,
                       events[i].is_timing_event ? "yes" : "no");
            }
        }

        printf("Active events: %d/%d\n", active_events, max_events);
        printf("Timing events: %d\n", timing_events);
        printf("Total usage count: %d\n", total_usage);
        printf("===============================\n");
    }

    // Destroy specific event
    void destroy_event(const std::string& name) {
        auto it = event_name_map.find(name);
        if (it != event_name_map.end()) {
            int event_id = it->second;

            if (events[event_id].event != nullptr) {
                cudaEventDestroy(events[event_id].event);
                events[event_id].event = nullptr;
                events[event_id].usage_count = 0;

                event_name_map.erase(it);

                printf("Destroyed event '%s' (ID: %d)\n", name.c_str(), event_id);
            }
        }
    }

    ~EventManager() {
        printf("Destroying EventManager...\n");

        for (int i = 0; i < max_events; i++) {
            if (events[i].event != nullptr) {
                cudaEventDestroy(events[i].event);
                printf("Destroyed event '%s'\n", events[i].name.c_str());
            }
        }

        printf("EventManager cleanup complete\n");
    }
};

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
