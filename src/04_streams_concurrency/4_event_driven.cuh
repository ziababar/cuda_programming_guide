#ifndef EVENT_DRIVEN_H
#define EVENT_DRIVEN_H

#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <queue>
#include <chrono>
#include <functional>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

// Forward declarations
__global__ void event_demo_kernel(float* data, int N, int kernel_id);
__global__ void complex_math_kernel(float* input, float* output, int N);
__global__ void simple_math_kernel(float* input, float* output, int N);
__global__ void initialization_kernel(float* data, int N);
__global__ void combine_kernel(float* data1, float* data2, float* output, int N);


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

// Sophisticated timing system using CUDA events
class PerformanceProfiler {
private:
    struct TimingRegion {
        std::string name;
        cudaEvent_t start_event;
        cudaEvent_t stop_event;
        std::vector<float> measurements;
        float total_time;
        int measurement_count;
        bool is_active;
    };

    std::map<std::string, TimingRegion> timing_regions;
    std::vector<std::string> active_regions;
    cudaStream_t profiling_stream;

public:
    PerformanceProfiler(cudaStream_t stream = 0) : profiling_stream(stream) {
        printf("PerformanceProfiler initialized\n");
    }

    // Create a new timing region
    void create_timing_region(const std::string& name) {
        if (timing_regions.find(name) != timing_regions.end()) {
            printf("Timing region '%s' already exists\n", name.c_str());
            return;
        }

        TimingRegion region;
        region.name = name;
        region.total_time = 0.0f;
        region.measurement_count = 0;
        region.is_active = false;

        // Create events for this region
        cudaEventCreate(&region.start_event);
        cudaEventCreate(&region.stop_event);

        timing_regions[name] = region;
        printf("Created timing region '%s'\n", name.c_str());
    }

    // Start timing a region
    void start_timing(const std::string& name) {
        auto it = timing_regions.find(name);
        if (it == timing_regions.end()) {
            create_timing_region(name);
            it = timing_regions.find(name);
        }

        if (it->second.is_active) {
            printf("Warning: Timing region '%s' is already active\n", name.c_str());
            return;
        }

        cudaEventRecord(it->second.start_event, profiling_stream);
        it->second.is_active = true;
        active_regions.push_back(name);

        printf("Started timing '%s'\n", name.c_str());
    }

    // Stop timing a region
    float stop_timing(const std::string& name) {
        auto it = timing_regions.find(name);
        if (it == timing_regions.end() || !it->second.is_active) {
            printf("Error: Timing region '%s' is not active\n", name.c_str());
            return -1.0f;
        }

        cudaEventRecord(it->second.stop_event, profiling_stream);
        cudaEventSynchronize(it->second.stop_event);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, it->second.start_event, it->second.stop_event);

        // Update statistics
        it->second.measurements.push_back(elapsed_time);
        it->second.total_time += elapsed_time;
        it->second.measurement_count++;
        it->second.is_active = false;

        // Remove from active regions
        active_regions.erase(
            std::remove(active_regions.begin(), active_regions.end(), name),
            active_regions.end());

        printf("Stopped timing '%s': %.3f ms\n", name.c_str(), elapsed_time);
        return elapsed_time;
    }

    // RAII-style timing guard
    class TimingGuard {
    private:
        PerformanceProfiler* profiler;
        std::string region_name;

    public:
        TimingGuard(PerformanceProfiler* prof, const std::string& name)
            : profiler(prof), region_name(name) {
            profiler->start_timing(region_name);
        }

        ~TimingGuard() {
            profiler->stop_timing(region_name);
        }
    };

    // Create timing guard for automatic scope-based timing
    TimingGuard time_scope(const std::string& name) {
        return TimingGuard(this, name);
    }

    // Get statistics for a timing region
    void print_region_stats(const std::string& name) {
        auto it = timing_regions.find(name);
        if (it == timing_regions.end()) {
            printf("Timing region '%s' not found\n", name.c_str());
            return;
        }

        const TimingRegion& region = it->second;

        if (region.measurement_count == 0) {
            printf("No measurements for region '%s'\n", name.c_str());
            return;
        }

        float avg_time = region.total_time / region.measurement_count;

        // Calculate min, max, and std deviation
        float min_time = *std::min_element(region.measurements.begin(), region.measurements.end());
        float max_time = *std::max_element(region.measurements.begin(), region.measurements.end());

        float variance = 0.0f;
        for (float measurement : region.measurements) {
            variance += (measurement - avg_time) * (measurement - avg_time);
        }
        variance /= region.measurement_count;
        float std_dev = sqrt(variance);

        printf("=== Timing Statistics for '%s' ===\n", name.c_str());
        printf("Measurements: %d\n", region.measurement_count);
        printf("Total time: %.3f ms\n", region.total_time);
        printf("Average time: %.3f ms\n", avg_time);
        printf("Min time: %.3f ms\n", min_time);
        printf("Max time: %.3f ms\n", max_time);
        printf("Std deviation: %.3f ms\n", std_dev);
        printf("Coefficient of variation: %.2f%%\n", (std_dev / avg_time) * 100.0f);
        printf("=====================================\n");
    }

    // Print all timing statistics
    void print_all_stats() {
        printf("=== Performance Profiler Summary ===\n");
        printf("Total timing regions: %zu\n", timing_regions.size());
        printf("Active regions: %zu\n", active_regions.size());

        if (!active_regions.empty()) {
            printf("Currently active: ");
            for (const auto& name : active_regions) {
                printf("'%s' ", name.c_str());
            }
            printf("\n");
        }

        printf("\nDetailed Statistics:\n");
        for (const auto& pair : timing_regions) {
            print_region_stats(pair.first);
        }
    }

    // Benchmark a specific operation multiple times
    void benchmark_operation(const std::string& name, std::function<void()> operation,
                           int iterations = 10) {
        printf("Benchmarking '%s' for %d iterations...\n", name.c_str(), iterations);

        for (int i = 0; i < iterations; i++) {
            start_timing(name);
            operation();
            stop_timing(name);
        }

        print_region_stats(name);
    }

    ~PerformanceProfiler() {
        printf("Destroying PerformanceProfiler...\n");

        // Stop any active timing regions
        for (const auto& name : active_regions) {
            printf("Warning: Timing region '%s' was still active\n", name.c_str());
        }

        // Destroy all events
        for (auto& pair : timing_regions) {
            cudaEventDestroy(pair.second.start_event);
            cudaEventDestroy(pair.second.stop_event);
        }

        printf("PerformanceProfiler cleanup complete\n");
    }
};


// Complex event-driven coordination system
class EventCoordinator {
private:
    struct DependencyNode {
        std::string name;
        cudaEvent_t completion_event;
        std::vector<std::string> dependencies;
        std::vector<std::string> dependents;
        std::function<void(cudaStream_t)> work_function;
        cudaStream_t assigned_stream;
        bool is_completed;
        bool is_scheduled;
    };

    std::map<std::string, DependencyNode> nodes;
    std::vector<cudaStream_t> stream_pool;
    std::queue<std::string> ready_queue;

public:
    EventCoordinator(const std::vector<cudaStream_t>& streams) : stream_pool(streams) {
        printf("EventCoordinator initialized with %zu streams\n", streams.size());
    }

    // Add a work node with dependencies
    void add_node(const std::string& name,
                  const std::vector<std::string>& dependencies,
                  std::function<void(cudaStream_t)> work_func) {

        if (nodes.find(name) != nodes.end()) {
            printf("Error: Node '%s' already exists\n", name.c_str());
            return;
        }

        DependencyNode node;
        node.name = name;
        node.dependencies = dependencies;
        node.work_function = work_func;
        node.is_completed = false;
        node.is_scheduled = false;
        node.assigned_stream = nullptr;

        // Create completion event
        cudaEventCreate(&node.completion_event);

        // Update dependent relationships
        for (const auto& dep : dependencies) {
            if (nodes.find(dep) != nodes.end()) {
                nodes[dep].dependents.push_back(name);
            }
        }

        nodes[name] = node;

        printf("Added node '%s' with %zu dependencies\n", name.c_str(), dependencies.size());
    }

    // Execute the dependency graph
    void execute_graph() {
        printf("=== Executing Dependency Graph ===\n");

        // Find nodes with no dependencies (ready to execute)
        for (auto& pair : nodes) {
            if (pair.second.dependencies.empty()) {
                ready_queue.push(pair.first);
                printf("Node '%s' ready for execution (no dependencies)\n", pair.first.c_str());
            }
        }

        int completed_nodes = 0;
        int total_nodes = nodes.size();

        while (completed_nodes < total_nodes) {
            // Schedule ready nodes
            while (!ready_queue.empty() && !stream_pool.empty()) {
                std::string node_name = ready_queue.front();
                ready_queue.pop();

                schedule_node(node_name);
            }

            // Check for completed nodes
            check_completions();

            // Small delay to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        printf("Dependency graph execution complete\n");
    }

private:
    void schedule_node(const std::string& node_name) {
        auto& node = nodes[node_name];

        if (node.is_scheduled) {
            return;
        }

        // Assign stream
        node.assigned_stream = stream_pool[0]; // Simple round-robin
        std::rotate(stream_pool.begin(), stream_pool.begin() + 1, stream_pool.end());

        // Wait for all dependencies
        for (const auto& dep_name : node.dependencies) {
            const auto& dep_node = nodes[dep_name];
            cudaStreamWaitEvent(node.assigned_stream, dep_node.completion_event, 0);
        }

        // Execute work
        printf("Executing node '%s' on stream %p\n", node_name.c_str(), node.assigned_stream);
        node.work_function(node.assigned_stream);

        // Record completion event
        cudaEventRecord(node.completion_event, node.assigned_stream);

        node.is_scheduled = true;
    }

    void check_completions() {
        for (auto& pair : nodes) {
            auto& node = pair.second;

            if (node.is_scheduled && !node.is_completed) {
                cudaError_t status = cudaEventQuery(node.completion_event);

                if (status == cudaSuccess) {
                    node.is_completed = true;
                    printf("Node '%s' completed\n", pair.first.c_str());

                    // Check if any dependents are now ready
                    for (const auto& dependent_name : node.dependents) {
                        if (is_node_ready(dependent_name)) {
                            ready_queue.push(dependent_name);
                            printf("Node '%s' now ready (dependencies satisfied)\n",
                                   dependent_name.c_str());
                        }
                    }
                }
            }
        }
    }

    bool is_node_ready(const std::string& node_name) {
        const auto& node = nodes[node_name];

        if (node.is_scheduled) {
            return false;
        }

        // Check if all dependencies are completed
        for (const auto& dep_name : node.dependencies) {
            if (!nodes[dep_name].is_completed) {
                return false;
            }
        }

        return true;
    }

public:
    // Print graph structure
    void print_graph_structure() {
        printf("=== Dependency Graph Structure ===\n");

        for (const auto& pair : nodes) {
            const auto& node = pair.second;

            printf("Node '%s':\n", pair.first.c_str());

            if (!node.dependencies.empty()) {
                printf("  Dependencies: ");
                for (const auto& dep : node.dependencies) {
                    printf("'%s' ", dep.c_str());
                }
                printf("\n");
            }

            if (!node.dependents.empty()) {
                printf("  Dependents: ");
                for (const auto& dep : node.dependents) {
                    printf("'%s' ", dep.c_str());
                }
                printf("\n");
            }

            printf("  Status: %s\n",
                   node.is_completed ? "Completed" :
                   (node.is_scheduled ? "Scheduled" : "Waiting"));
        }

        printf("===============================\n");
    }

    ~EventCoordinator() {
        printf("Destroying EventCoordinator...\n");

        for (auto& pair : nodes) {
            cudaEventDestroy(pair.second.completion_event);
        }

        printf("EventCoordinator cleanup complete\n");
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

__global__ void combine_kernel(float* data1, float* data2, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = (data1[tid] + data2[tid]) * 0.5f;
    }
}

#endif // EVENT_DRIVEN_H
