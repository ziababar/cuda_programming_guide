#  CUDA Graphs Deep Dive

CUDA Graphs represent a paradigm shift from dynamic kernel launches to static execution graphs, enabling dramatic performance improvements for repetitive workloads by reducing launch overhead and enabling advanced optimizations.

##  Graph Fundamentals and Architecture

CUDA Graphs capture sequences of GPU operations into a static directed acyclic graph (DAG), allowing the CUDA runtime to optimize execution and minimize overhead.

###  Comprehensive Graph Management System
See [GraphManager.h](../src/04_streams_concurrency/GraphManager.h) for the full implementation of the `GraphManager` class.

```cpp
// Demonstrate basic graph creation and execution
void demonstrate_basic_graph_operations() {
    printf("=== Basic Graph Operations Demo ===\n");

    GraphManager manager;

    // Create test data
    const int N = 1024 * 1024;
    float *h_input, *h_output, *d_input, *d_output, *d_temp;

    cudaHostAlloc(&h_input, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_output, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = sin(i * 0.001f);
    }

    // Create capture stream
    cudaStream_t capture_stream = manager.create_capture_stream("main_stream");

    printf("\n1. Creating Basic Graph:\n");

    // Begin graph capture
    manager.begin_capture("basic_pipeline", "main_stream");

    // Enqueue operations to be captured
    cudaMemcpyAsync(d_input, h_input, N * sizeof(float),
                   cudaMemcpyHostToDevice, capture_stream);

    graph_stage1_kernel<<<(N+255)/256, 256, 0, capture_stream>>>(d_input, d_temp, N);
    graph_stage2_kernel<<<(N+255)/256, 256, 0, capture_stream>>>(d_temp, d_output, N);

    cudaMemcpyAsync(h_output, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, capture_stream);

    // End capture
    manager.end_capture("basic_pipeline", "main_stream");

    // Instantiate the graph
    manager.instantiate_graph("basic_pipeline");

    printf("\n2. Comparing Performance: Stream vs Graph:\n");

    // Measure stream-based execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_iterations = 100;

    // Stream execution timing
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        cudaMemcpyAsync(d_input, h_input, N * sizeof(float),
                       cudaMemcpyHostToDevice, capture_stream);
        graph_stage1_kernel<<<(N+255)/256, 256, 0, capture_stream>>>(d_input, d_temp, N);
        graph_stage2_kernel<<<(N+255)/256, 256, 0, capture_stream>>>(d_temp, d_output, N);
        cudaMemcpyAsync(h_output, d_output, N * sizeof(float),
                       cudaMemcpyDeviceToHost, capture_stream);
    }
    cudaStreamSynchronize(capture_stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float stream_time;
    cudaEventElapsedTime(&stream_time, start, stop);

    // Graph execution timing
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        manager.launch_graph("basic_pipeline", capture_stream);
    }
    cudaStreamSynchronize(capture_stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float graph_time;
    cudaEventElapsedTime(&graph_time, start, stop);

    printf("Stream execution time: %.3f ms\n", stream_time);
    printf("Graph execution time: %.3f ms\n", graph_time);
    printf("Speedup: %.2fx\n", stream_time / graph_time);
    printf("Launch overhead reduction: %.2f%%\n",
           ((stream_time - graph_time) / stream_time) * 100.0f);

    // Print statistics
    manager.print_graph_statistics("basic_pipeline");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

__global__ void graph_stage1_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = sqrt(fabs(input[tid])) + sin(input[tid]);
    }
}

__global__ void graph_stage2_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = cos(input[tid]) + log(fabs(input[tid]) + 1.0f);
    }
}
```

##  Advanced Graph Patterns and Optimization

###  Dynamic Graph Updates and Parameter Modification
```cpp
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

// Kernel implementations for graph patterns
__global__ void parameterized_kernel(float* input, float* output, int N, float scale) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = input[tid] * scale + sin(tid * 0.001f);
    }
}

__global__ void parallel_branch_kernel(float* input, float* output, int N, int branch_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float value = input[tid];

        // Different processing per branch
        if (branch_id == 1) {
            value = sqrt(fabs(value)) + cos(value);
        } else {
            value = sin(value) + log(fabs(value) + 1.0f);
        }

        output[tid] = value;
    }
}

__global__ void combine_results_kernel(float* input1, float* input2, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = (input1[tid] + input2[tid]) * 0.5f;
    }
}

__global__ void preprocessing_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = input[tid] * 2.0f + 1.0f;
    }
}

__global__ void optimized_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Optimized computation (fewer operations)
        output[tid] = input[tid] + 0.5f;
    }
}

__global__ void standard_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Standard computation (more operations)
        float value = input[tid];
        for (int i = 0; i < 10; i++) {
            value = sin(value) + cos(value * 0.1f);
        }
        output[tid] = value;
    }
}

__global__ void postprocessing_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = sqrt(fabs(input[tid]));
    }
}
```

##  Production Graph Optimization Strategies

###  Enterprise-Grade Graph Management
```cpp
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

__global__ void batched_operation_kernel(float* input, float* output, int N, int operation_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = input[tid];

        // Different operations based on ID
        switch (operation_id % 4) {
            case 0:
                value = sin(value) + 1.0f;
                break;
            case 1:
                value = cos(value) + 2.0f;
                break;
            case 2:
                value = sqrt(fabs(value)) + 3.0f;
                break;
            case 3:
                value = log(fabs(value) + 1.0f) + 4.0f;
                break;
        }

        output[tid] = value;
    }
}
```
