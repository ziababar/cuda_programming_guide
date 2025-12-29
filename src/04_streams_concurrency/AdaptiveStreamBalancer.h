#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <numeric>
#include <cstdio>
#include <algorithm>

// Forward declarations for kernels
__global__ void light_compute_kernel(int task_id);
__global__ void medium_compute_kernel(int task_id);
__global__ void heavy_compute_kernel(int task_id);

// Dynamic load balancing across multiple streams
class AdaptiveStreamBalancer {
private:
    struct StreamWorker {
        cudaStream_t stream;
        std::string name;
        float avg_processing_time;
        int completed_tasks;
        int queued_tasks;
        float load_factor;
        bool is_active;
        cudaEvent_t last_completion;
    };

    std::vector<StreamWorker> workers;
    std::queue<int> task_queue;
    std::mutex balancer_mutex;
    std::condition_variable task_available;

    int num_workers;
    bool balancer_active;
    std::atomic<int> total_completed_tasks;
    std::atomic<int> total_submitted_tasks;

    // Performance monitoring
    std::chrono::high_resolution_clock::time_point start_time;
    float total_processing_time;

public:
    AdaptiveStreamBalancer(int num_stream_workers)
        : num_workers(num_stream_workers), balancer_active(true),
          total_completed_tasks(0), total_submitted_tasks(0), total_processing_time(0.0f) {

        printf("Initializing AdaptiveStreamBalancer with %d workers\n", num_workers);

        workers.resize(num_workers);

        // Initialize stream workers
        for (int i = 0; i < num_workers; i++) {
            StreamWorker& worker = workers[i];
            worker.name = "Worker_" + std::to_string(i);

            cudaStreamCreate(&worker.stream);
            cudaEventCreate(&worker.last_completion);

            worker.avg_processing_time = 0.0f;
            worker.completed_tasks = 0;
            worker.queued_tasks = 0;
            worker.load_factor = 0.0f;
            worker.is_active = true;

            printf("Initialized %s (stream: %p)\n", worker.name.c_str(), worker.stream);
        }

        start_time = std::chrono::high_resolution_clock::now();
        printf("AdaptiveStreamBalancer initialization complete\n");
    }

    // Submit a task for processing
    int submit_task(std::function<void(cudaStream_t, int)> task_func, int task_data) {
        std::lock_guard<std::mutex> lock(balancer_mutex);

        // Find the worker with the lowest load
        int best_worker = select_optimal_worker();

        if (best_worker < 0) {
            printf("No available workers\n");
            return -1;
        }

        // Execute task on selected worker
        StreamWorker& worker = workers[best_worker];

        // Record start time
        cudaEvent_t task_start;
        cudaEventCreate(&task_start);
        cudaEventRecord(task_start, worker.stream);

        // Execute the task
        task_func(worker.stream, task_data);

        // Record completion
        cudaEventRecord(worker.last_completion, worker.stream);

        // Update worker statistics (asynchronously)
        worker.queued_tasks++;
        total_submitted_tasks++;

        // Schedule statistics update
        std::thread([this, best_worker, task_start]() {
            update_worker_statistics(best_worker, task_start);
        }).detach();

        printf("Submitted task %d to %s (load: %.2f)\n",
               task_data, worker.name.c_str(), worker.load_factor);

        return best_worker;
    }

    // Batch submit multiple tasks
    void submit_task_batch(const std::vector<std::function<void(cudaStream_t, int)>>& tasks,
                          const std::vector<int>& task_data) {
        printf("Submitting batch of %zu tasks\n", tasks.size());

        for (size_t i = 0; i < tasks.size(); i++) {
            int worker_id = submit_task(tasks[i], task_data[i]);

            if (worker_id < 0) {
                printf("Failed to submit task %zu\n", i);
                break;
            }

            // Small delay to allow load balancing to adapt
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        printf("Batch submission complete\n");
    }

    // Wait for all tasks to complete
    void wait_for_completion() {
        printf("Waiting for all tasks to complete...\n");

        for (auto& worker : workers) {
            cudaStreamSynchronize(worker.stream);
        }

        printf("All tasks completed\n");
    }

    // Rebalance load based on performance metrics
    void rebalance_load() {
        std::lock_guard<std::mutex> lock(balancer_mutex);

        printf("=== Load Rebalancing ===\n");

        // Calculate total load
        float total_load = 0.0f;
        int active_workers = 0;

        for (const auto& worker : workers) {
            if (worker.is_active) {
                total_load += worker.load_factor;
                active_workers++;
            }
        }

        if (active_workers == 0) {
            printf("No active workers available\n");
            return;
        }

        float avg_load = total_load / active_workers;
        printf("Average load per worker: %.3f\n", avg_load);

        // Identify overloaded and underloaded workers
        std::vector<int> overloaded_workers;
        std::vector<int> underloaded_workers;

        const float load_threshold = 1.2f; // 20% above average

        for (int i = 0; i < num_workers; i++) {
            if (!workers[i].is_active) continue;

            float load_ratio = workers[i].load_factor / avg_load;

            if (load_ratio > load_threshold) {
                overloaded_workers.push_back(i);
                printf("  %s is overloaded (ratio: %.2f)\n",
                       workers[i].name.c_str(), load_ratio);
            } else if (load_ratio < (1.0f / load_threshold)) {
                underloaded_workers.push_back(i);
                printf("  %s is underloaded (ratio: %.2f)\n",
                       workers[i].name.c_str(), load_ratio);
            }
        }

        // Implement load balancing strategy
        if (!overloaded_workers.empty() && !underloaded_workers.empty()) {
            printf("Load balancing opportunities identified\n");
            // In a real implementation, we could migrate queued tasks
            // or adjust worker priorities
        }

        printf("Load rebalancing complete\n");
    }

    // Print comprehensive statistics
    void print_comprehensive_statistics() {
        printf("=== Comprehensive Load Balancer Statistics ===\n");

        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time);

        printf("Runtime: %ld ms\n", elapsed.count());
        printf("Total tasks submitted: %d\n", total_submitted_tasks.load());
        printf("Total tasks completed: %d\n", total_completed_tasks.load());

        float completion_rate = (total_completed_tasks.load() > 0) ?
            (total_completed_tasks.load() * 100.0f) / total_submitted_tasks.load() : 0.0f;
        printf("Completion rate: %.1f%%\n", completion_rate);

        if (elapsed.count() > 0) {
            float throughput = (total_completed_tasks.load() * 1000.0f) / elapsed.count();
            printf("Throughput: %.2f tasks/second\n", throughput);
        }

        printf("\nPer-Worker Statistics:\n");

        float total_avg_time = 0.0f;
        int active_workers = 0;

        for (int i = 0; i < num_workers; i++) {
            const StreamWorker& worker = workers[i];

            printf("  %s:\n", worker.name.c_str());
            printf("    Status: %s\n", worker.is_active ? "Active" : "Inactive");
            printf("    Completed tasks: %d\n", worker.completed_tasks);
            printf("    Queued tasks: %d\n", worker.queued_tasks);
            printf("    Average processing time: %.3f ms\n", worker.avg_processing_time);
            printf("    Load factor: %.3f\n", worker.load_factor);

            if (worker.is_active && worker.completed_tasks > 0) {
                total_avg_time += worker.avg_processing_time;
                active_workers++;
            }
        }

        if (active_workers > 0) {
            printf("\nOverall average processing time: %.3f ms\n",
                   total_avg_time / active_workers);
        }

        // Load distribution analysis
        printf("\nLoad Distribution Analysis:\n");
        analyze_load_distribution();

        printf("=============================================\n");
    }

private:
    int select_optimal_worker() {
        int best_worker = -1;
        float lowest_load = std::numeric_limits<float>::max();

        for (int i = 0; i < num_workers; i++) {
            if (!workers[i].is_active) continue;

            // Calculate current load considering queued tasks and processing time
            float current_load = workers[i].load_factor +
                               (workers[i].queued_tasks * workers[i].avg_processing_time);

            if (current_load < lowest_load || best_worker < 0) {
                lowest_load = current_load;
                best_worker = i;
            }
        }

        return best_worker;
    }

    void update_worker_statistics(int worker_id, cudaEvent_t task_start) {
        if (worker_id < 0 || worker_id >= num_workers) return;

        StreamWorker& worker = workers[worker_id];

        // Wait for task completion
        cudaEventSynchronize(worker.last_completion);

        // Measure task execution time
        float task_time;
        cudaEventElapsedTime(&task_time, task_start, worker.last_completion);

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(balancer_mutex);

            worker.avg_processing_time =
                (worker.avg_processing_time * worker.completed_tasks + task_time) /
                (worker.completed_tasks + 1);

            worker.completed_tasks++;
            worker.queued_tasks--;

            // Update load factor (simple heuristic based on processing time and queue length)
            worker.load_factor = worker.avg_processing_time *
                               (1.0f + worker.queued_tasks * 0.1f);

            total_completed_tasks++;
            total_processing_time += task_time;
        }

        cudaEventDestroy(task_start);
    }

    void analyze_load_distribution() {
        if (num_workers == 0) return;

        // Calculate load distribution metrics
        std::vector<float> loads;
        for (const auto& worker : workers) {
            if (worker.is_active) {
                loads.push_back(worker.load_factor);
            }
        }

        if (loads.empty()) {
            printf("  No active workers to analyze\n");
            return;
        }

        // Calculate statistics
        float sum = std::accumulate(loads.begin(), loads.end(), 0.0f);
        float mean = sum / loads.size();

        float min_load = *std::min_element(loads.begin(), loads.end());
        float max_load = *std::max_element(loads.begin(), loads.end());

        // Calculate standard deviation
        float variance = 0.0f;
        for (float load : loads) {
            variance += (load - mean) * (load - mean);
        }
        variance /= loads.size();
        float std_dev = sqrt(variance);

        printf("  Load mean: %.3f\n", mean);
        printf("  Load range: %.3f - %.3f\n", min_load, max_load);
        printf("  Load std deviation: %.3f\n", std_dev);
        printf("  Load balance coefficient: %.2f%% (lower is better)\n",
               (std_dev / mean) * 100.0f);

        // Classification
        float balance_ratio = std_dev / mean;
        if (balance_ratio < 0.1f) {
            printf("  Balance quality: Excellent\n");
        } else if (balance_ratio < 0.2f) {
            printf("  Balance quality: Good\n");
        } else if (balance_ratio < 0.3f) {
            printf("  Balance quality: Fair\n");
        } else {
            printf("  Balance quality: Poor - rebalancing recommended\n");
        }
    }

public:
    ~AdaptiveStreamBalancer() {
        printf("Destroying AdaptiveStreamBalancer...\n");

        balancer_active = false;

        // Wait for all tasks to complete
        wait_for_completion();

        // Cleanup resources
        for (auto& worker : workers) {
            cudaStreamDestroy(worker.stream);
            cudaEventDestroy(worker.last_completion);
        }

        printf("AdaptiveStreamBalancer cleanup complete\n");
        printf("Final statistics: %d tasks completed\n", total_completed_tasks.load());
    }
};

// Load balancing kernel implementations
__global__ void light_compute_kernel(int task_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Light computation
    float result = 0.0f;
    for (int i = 0; i < 100; i++) {
        result += sin(tid * 0.001f + i);
    }

    if (tid == 0) {
        printf("Light task %d completed (result: %.3f)\n", task_id, result);
    }
}

__global__ void medium_compute_kernel(int task_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Medium computation
    float result = 0.0f;
    for (int i = 0; i < 1000; i++) {
        result += sin(tid * 0.001f + i) * cos(tid * 0.002f + i);
    }

    if (tid == 0) {
        printf("Medium task %d completed (result: %.3f)\n", task_id, result);
    }
}

__global__ void heavy_compute_kernel(int task_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Heavy computation
    float result = 0.0f;
    for (int i = 0; i < 10000; i++) {
        result += sin(tid * 0.001f + i) * cos(tid * 0.002f + i) *
                 log(fabs(tid * 0.003f + i) + 1.0f);
    }

    if (tid == 0) {
        printf("Heavy task %d completed (result: %.3f)\n", task_id, result);
    }
}
