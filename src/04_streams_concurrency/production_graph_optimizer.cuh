#ifndef PRODUCTION_GRAPH_OPTIMIZER_CUH
#define PRODUCTION_GRAPH_OPTIMIZER_CUH

#include <vector>
#include <map>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "graph_manager.cuh"

// Forward declaration of kernels
__global__ void batched_operation_kernel(float* input, float* output, int N, int operation_id);

// Production-ready graph optimization and management
class ProductionGraphOptimizer {
private:
    GraphManager* graph_manager;
    std::map<std::string, std::vector<float>> performance_history;
    std::map<std::string, int> usage_frequency;

public:
    ProductionGraphOptimizer(GraphManager* manager) : graph_manager(manager) {
        printf("ProductionGraphOptimizer initialized\n");
    }

    // Optimize graph execution order based on performance
    void optimize_execution_order(const std::vector<std::string>& graph_names,
                                 int optimization_iterations = 100) {

        printf("=== Graph Execution Order Optimization ===\n");

        if (graph_names.size() < 2) {
            printf("Need at least 2 graphs for optimization\n");
            return;
        }

        // Measure individual graph performance
        std::map<std::string, float> individual_times;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (const auto& graph_name : graph_names) {
            cudaEventRecord(start);
            for (int i = 0; i < 10; i++) {
                graph_manager->launch_graph(graph_name);
            }
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time;
            cudaEventElapsedTime(&time, start, stop);
            individual_times[graph_name] = time / 10.0f;

            printf("Graph '%s' individual time: %.3f ms\n", graph_name.c_str(), time / 10.0f);
        }

        // Test different execution orders
        std::vector<std::string> best_order = graph_names;
        float best_time = std::numeric_limits<float>::max();

        // Try different permutations
        std::vector<std::string> current_order = graph_names;

        do {
            // Measure this execution order
            cudaEventRecord(start);
            for (int iter = 0; iter < 5; iter++) {
                for (const auto& graph_name : current_order) {
                    graph_manager->launch_graph(graph_name);
                }
            }
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float total_time;
            cudaEventElapsedTime(&total_time, start, stop);
            total_time /= 5.0f; // Average per iteration

            if (total_time < best_time) {
                best_time = total_time;
                best_order = current_order;
            }

        } while (std::next_permutation(current_order.begin(), current_order.end()) &&
                 current_order.size() <= 4); // Limit permutations for large sets

        printf("\nOptimization Results:\n");
        printf("Best execution order: ");
        for (const auto& name : best_order) {
            printf("'%s' ", name.c_str());
        }
        printf("\n");
        printf("Best total time: %.3f ms\n", best_time);

        // Calculate efficiency
        float sum_individual = 0.0f;
        for (const auto& pair : individual_times) {
            sum_individual += pair.second;
        }

        printf("Sum of individual times: %.3f ms\n", sum_individual);
        printf("Optimization efficiency: %.2f%%\n",
               ((sum_individual - best_time) / sum_individual) * 100.0f);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Create optimized graph batches
    void create_batched_graph(const std::string& batch_name,
                             const std::vector<std::string>& component_graphs) {

        printf("Creating batched graph '%s' from %zu components\n",
               batch_name.c_str(), component_graphs.size());

        cudaStream_t batch_stream = graph_manager->create_capture_stream(batch_name + "_stream");

        // Begin capture for the batch
        graph_manager->begin_capture(batch_name, batch_name + "_stream");

        // Execute all component operations in sequence
        // Note: This is a simplified example - real batching would need more sophisticated
        // graph composition techniques

        // For demonstration, we'll create a synthetic batch
        const int N = 1024 * 1024;
        float *d_temp1, *d_temp2;
        cudaMalloc(&d_temp1, N * sizeof(float));
        cudaMalloc(&d_temp2, N * sizeof(float));

        // Simulate batched operations
        for (size_t i = 0; i < component_graphs.size(); i++) {
            batched_operation_kernel<<<(N+255)/256, 256, 0, batch_stream>>>(
                d_temp1, d_temp2, N, static_cast<int>(i));
        }

        // End capture
        graph_manager->end_capture(batch_name, batch_name + "_stream");

        // Instantiate
        bool success = graph_manager->instantiate_graph(batch_name);

        if (success) {
            printf("Batched graph '%s' created successfully\n", batch_name.c_str());
        }

        // Cleanup temporary memory
        cudaFree(d_temp1);
        cudaFree(d_temp2);
    }

    // Analyze graph performance patterns
    void analyze_performance_patterns(const std::vector<std::string>& graph_names,
                                     int analysis_iterations = 50) {

        printf("=== Performance Pattern Analysis ===\n");

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Collect performance data
        for (const auto& graph_name : graph_names) {
            std::vector<float> execution_times;

            for (int i = 0; i < analysis_iterations; i++) {
                cudaEventRecord(start);
                graph_manager->launch_graph(graph_name);
                cudaDeviceSynchronize();
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float time;
                cudaEventElapsedTime(&time, start, stop);
                execution_times.push_back(time);
            }

            // Calculate statistics
            float min_time = *std::min_element(execution_times.begin(), execution_times.end());
            float max_time = *std::max_element(execution_times.begin(), execution_times.end());

            float avg_time = 0.0f;
            for (float time : execution_times) {
                avg_time += time;
            }
            avg_time /= execution_times.size();

            float variance = 0.0f;
            for (float time : execution_times) {
                variance += (time - avg_time) * (time - avg_time);
            }
            variance /= execution_times.size();
            float std_dev = sqrt(variance);

            // Store performance history
            performance_history[graph_name] = execution_times;

            printf("\nGraph '%s' Performance Analysis (%d iterations):\n",
                   graph_name.c_str(), analysis_iterations);
            printf("  Min time: %.3f ms\n", min_time);
            printf("  Max time: %.3f ms\n", max_time);
            printf("  Average time: %.3f ms\n", avg_time);
            printf("  Std deviation: %.3f ms\n", std_dev);
            printf("  Coefficient of variation: %.2f%%\n", (std_dev / avg_time) * 100.0f);

            // Performance stability assessment
            if ((std_dev / avg_time) < 0.05f) {
                printf("  Assessment: Highly stable performance\n");
            } else if ((std_dev / avg_time) < 0.15f) {
                printf("  Assessment: Stable performance\n");
            } else {
                printf("  Assessment: Variable performance - investigate optimization\n");
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Generate performance report
    void generate_performance_report() {
        printf("\n=== Comprehensive Performance Report ===\n");

        if (performance_history.empty()) {
            printf("No performance data available. Run analysis first.\n");
            return;
        }

        printf("Graphs analyzed: %zu\n", performance_history.size());

        // Find best and worst performing graphs
        std::string fastest_graph, slowest_graph;
        float fastest_time = std::numeric_limits<float>::max();
        float slowest_time = 0.0f;

        for (const auto& pair : performance_history) {
            const auto& times = pair.second;
            float avg_time = 0.0f;
            for (float time : times) {
                avg_time += time;
            }
            avg_time /= times.size();

            if (avg_time < fastest_time) {
                fastest_time = avg_time;
                fastest_graph = pair.first;
            }

            if (avg_time > slowest_time) {
                slowest_time = avg_time;
                slowest_graph = pair.first;
            }
        }

        printf("\nPerformance Summary:\n");
        printf("Fastest graph: '%s' (%.3f ms average)\n", fastest_graph.c_str(), fastest_time);
        printf("Slowest graph: '%s' (%.3f ms average)\n", slowest_graph.c_str(), slowest_time);
        printf("Performance ratio: %.2fx\n", slowest_time / fastest_time);

        // Optimization recommendations
        printf("\nOptimization Recommendations:\n");
        for (const auto& pair : performance_history) {
            const auto& times = pair.second;

            float avg_time = 0.0f;
            for (float time : times) {
                avg_time += time;
            }
            avg_time /= times.size();

            float variance = 0.0f;
            for (float time : times) {
                variance += (time - avg_time) * (time - avg_time);
            }
            variance /= times.size();
            float cv = sqrt(variance) / avg_time;

            printf("  '%s': ", pair.first.c_str());

            if (avg_time > fastest_time * 2.0f) {
                printf("Consider optimizing - significantly slower than best performer\n");
            } else if (cv > 0.15f) {
                printf("High variability - investigate resource contention\n");
            } else {
                printf("Performance acceptable\n");
            }
        }

        printf("========================================\n");
    }
};

#endif // PRODUCTION_GRAPH_OPTIMIZER_CUH
