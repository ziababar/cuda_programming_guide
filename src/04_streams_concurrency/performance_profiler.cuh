#ifndef PERFORMANCE_PROFILER_CUH
#define PERFORMANCE_PROFILER_CUH

#include <vector>
#include <string>
#include <map>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

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

#endif // PERFORMANCE_PROFILER_CUH
