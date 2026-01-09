#ifndef ADVANCED_GRAPH_PATTERNS_CUH
#define ADVANCED_GRAPH_PATTERNS_CUH

#include <string>
#include <map>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include "graph_manager.cuh"

// Forward declaration of kernels
__global__ void parameterized_kernel(float* input, float* output, int N, float scale);
__global__ void parallel_branch_kernel(float* input, float* output, int N, int branch_id);
__global__ void combine_results_kernel(float* input1, float* input2, float* output, int N);
__global__ void preprocessing_kernel(float* input, float* output, int N);
__global__ void optimized_kernel(float* input, float* output, int N);
__global__ void standard_kernel(float* input, float* output, int N);
__global__ void postprocessing_kernel(float* input, float* output, int N);

// Advanced graph patterns for production workloads
class AdvancedGraphPatterns {
private:
    GraphManager* graph_manager;
    std::map<std::string, std::vector<cudaGraphNode_t>> graph_nodes;

public:
    AdvancedGraphPatterns(GraphManager* manager) : graph_manager(manager) {
        printf("AdvancedGraphPatterns initialized\n");
    }

    // Create a parameterized graph that can be updated
    bool create_parameterized_graph(const std::string& graph_name,
                                   float* d_input, float* d_output,
                                   int N, float initial_scale = 1.0f) {

        cudaStream_t stream = graph_manager->create_capture_stream(graph_name + "_stream");

        // Begin capture
        graph_manager->begin_capture(graph_name, graph_name + "_stream");

        // Create kernels with parameters that can be updated
        parameterized_kernel<<<(N+255)/256, 256, 0, stream>>>(d_input, d_output, N, initial_scale);

        // End capture
        graph_manager->end_capture(graph_name, graph_name + "_stream");

        // Instantiate
        bool success = graph_manager->instantiate_graph(graph_name);

        if (success) {
            printf("Created parameterized graph '%s' with initial scale %.2f\n",
                   graph_name.c_str(), initial_scale);
        }

        return success;
    }

    // Create a multi-stream graph with dependencies
    bool create_multi_stream_graph(const std::string& graph_name,
                                  float* d_input1, float* d_input2,
                                  float* d_output1, float* d_output2,
                                  float* d_combined, int N) {

        printf("Creating multi-stream graph '%s'...\n", graph_name.c_str());

        // Create multiple streams for parallel execution
        cudaStream_t stream1 = graph_manager->create_capture_stream(graph_name + "_stream1");
        cudaStream_t stream2 = graph_manager->create_capture_stream(graph_name + "_stream2");
        cudaStream_t main_stream = graph_manager->create_capture_stream(graph_name + "_main");

        // Create events for synchronization
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);

        // Begin capture on main stream (captures entire graph)
        graph_manager->begin_capture(graph_name, graph_name + "_main", cudaStreamCaptureModeGlobal);

        // Parallel processing branches
        // Branch 1
        parallel_branch_kernel<<<(N+255)/256, 256, 0, stream1>>>(d_input1, d_output1, N, 1);
        cudaEventRecord(event1, stream1);

        // Branch 2
        parallel_branch_kernel<<<(N+255)/256, 256, 0, stream2>>>(d_input2, d_output2, N, 2);
        cudaEventRecord(event2, stream2);

        // Synchronization point
        cudaStreamWaitEvent(main_stream, event1, 0);
        cudaStreamWaitEvent(main_stream, event2, 0);

        // Combine results
        combine_results_kernel<<<(N+255)/256, 256, 0, main_stream>>>(
            d_output1, d_output2, d_combined, N);

        // End capture
        graph_manager->end_capture(graph_name, graph_name + "_main");

        // Instantiate
        bool success = graph_manager->instantiate_graph(graph_name);

        // Cleanup events
        cudaEventDestroy(event1);
        cudaEventDestroy(event2);

        if (success) {
            printf("Multi-stream graph '%s' created successfully\n", graph_name.c_str());
        }

        return success;
    }

    // Create a conditional execution graph
    bool create_conditional_graph(const std::string& graph_name,
                                 float* d_input, float* d_output,
                                 int N, bool enable_optimization = true) {

        printf("Creating conditional graph '%s'...\n", graph_name.c_str());

        cudaStream_t stream = graph_manager->create_capture_stream(graph_name + "_stream");

        graph_manager->begin_capture(graph_name, graph_name + "_stream");

        // Always execute preprocessing
        preprocessing_kernel<<<(N+255)/256, 256, 0, stream>>>(d_input, d_output, N);

        // Conditional execution based on parameter
        if (enable_optimization) {
            optimized_kernel<<<(N+255)/256, 256, 0, stream>>>(d_output, d_output, N);
            printf("  Including optimization kernel\n");
        } else {
            standard_kernel<<<(N+255)/256, 256, 0, stream>>>(d_output, d_output, N);
            printf("  Including standard kernel\n");
        }

        // Always execute postprocessing
        postprocessing_kernel<<<(N+255)/256, 256, 0, stream>>>(d_output, d_output, N);

        graph_manager->end_capture(graph_name, graph_name + "_stream");

        bool success = graph_manager->instantiate_graph(graph_name);

        if (success) {
            printf("Conditional graph '%s' created (optimization: %s)\n",
                   graph_name.c_str(), enable_optimization ? "enabled" : "disabled");
        }

        return success;
    }

    // Benchmark different graph patterns
    void benchmark_graph_patterns() {
        printf("=== Graph Patterns Benchmark ===\n");

        const int N = 2 * 1024 * 1024;
        const int iterations = 50;

        // Allocate test data
        float *d_input1, *d_input2, *d_output1, *d_output2, *d_combined;
        cudaMalloc(&d_input1, N * sizeof(float));
        cudaMalloc(&d_input2, N * sizeof(float));
        cudaMalloc(&d_output1, N * sizeof(float));
        cudaMalloc(&d_output2, N * sizeof(float));
        cudaMalloc(&d_combined, N * sizeof(float));

        // Initialize data
        cudaMemset(d_input1, 0, N * sizeof(float));
        cudaMemset(d_input2, 1, N * sizeof(float));

        // Create different graph patterns
        create_parameterized_graph("param_graph", d_input1, d_output1, N, 2.0f);
        create_multi_stream_graph("multi_stream_graph", d_input1, d_input2,
                                 d_output1, d_output2, d_combined, N);
        create_conditional_graph("conditional_optimized", d_input1, d_output1, N, true);
        create_conditional_graph("conditional_standard", d_input1, d_output1, N, false);

        // Benchmark each pattern
        std::vector<std::string> graph_names = {
            "param_graph", "multi_stream_graph",
            "conditional_optimized", "conditional_standard"
        };

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (const auto& graph_name : graph_names) {
            printf("\nBenchmarking '%s':\n", graph_name.c_str());

            cudaEventRecord(start);
            for (int i = 0; i < iterations; i++) {
                graph_manager->launch_graph(graph_name);
            }
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float total_time;
            cudaEventElapsedTime(&total_time, start, stop);

            printf("  %d iterations: %.3f ms total, %.3f ms average\n",
                   iterations, total_time, total_time / iterations);

            graph_manager->print_graph_statistics(graph_name);
        }

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_output1);
        cudaFree(d_output2);
        cudaFree(d_combined);
    }

    // Demonstrate graph cloning and modification
    void demonstrate_graph_cloning() {
        printf("=== Graph Cloning and Modification Demo ===\n");

        const int N = 1024 * 1024;
        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));

        // Create original graph
        create_parameterized_graph("original", d_input, d_output, N, 1.0f);

        // Clone the graph
        graph_manager->clone_graph("original", "cloned");
        graph_manager->instantiate_graph("cloned");

        // Execute both graphs and compare
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const int test_iterations = 20;

        // Original graph
        cudaEventRecord(start);
        for (int i = 0; i < test_iterations; i++) {
            graph_manager->launch_graph("original");
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float original_time;
        cudaEventElapsedTime(&original_time, start, stop);

        // Cloned graph
        cudaEventRecord(start);
        for (int i = 0; i < test_iterations; i++) {
            graph_manager->launch_graph("cloned");
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float cloned_time;
        cudaEventElapsedTime(&cloned_time, start, stop);

        printf("Original graph time: %.3f ms\n", original_time);
        printf("Cloned graph time: %.3f ms\n", cloned_time);
        printf("Performance difference: %.2f%%\n",
               ((cloned_time - original_time) / original_time) * 100.0f);

        // Print statistics for both
        graph_manager->print_graph_statistics("original");
        graph_manager->print_graph_statistics("cloned");

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_output);
    }
};

#endif // ADVANCED_GRAPH_PATTERNS_CUH
