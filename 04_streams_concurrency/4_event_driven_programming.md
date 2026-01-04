# Event-Driven Programming

CUDA events provide precise synchronization control and performance measurement capabilities, enabling sophisticated coordination between streams and accurate timing analysis of GPU operations.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Related: [Stream Fundamentals](1_stream_fundamentals.md)**

---

## Event Fundamentals and Types

CUDA events serve as lightweight synchronization primitives that can be recorded in streams and waited upon, providing fine-grained control over execution dependencies.

### Comprehensive Event Management

**Source Code**: [`../src/04_streams_concurrency/event_manager.h`](../src/04_streams_concurrency/event_manager.h)

```cpp
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
```

## Precision Timing and Performance Measurement

Events provide the most accurate method for measuring GPU execution times, with sub-millisecond precision and minimal overhead.

### Advanced Timing Infrastructure

```cpp
// Full implementation available in ../src/04_streams_concurrency/performance_profiler.cuh
#include "../src/04_streams_concurrency/performance_profiler.cuh"

// Comprehensive timing demonstration
void demonstrate_performance_profiling() {
    printf("=== Performance Profiling Demonstration ===\n");

    PerformanceProfiler profiler;

    // Create test data
    const int N = 2 * 1024 * 1024; // 2M elements
    float *h_data, *d_input, *d_output, *d_temp;

    cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = sin(i * 0.001f);
    }

    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    printf("\n1. Basic Operation Timing:\n");

    // Time memory transfer
    profiler.start_timing("memory_h2d");
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    profiler.stop_timing("memory_h2d");

    // Time kernel execution
    profiler.start_timing("kernel_execution");
    complex_math_kernel<<<(N+255)/256, 256>>>(d_input, d_output, N);
    profiler.stop_timing("kernel_execution");

    // Time memory transfer back
    profiler.start_timing("memory_d2h");
    cudaMemcpy(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    profiler.stop_timing("memory_d2h");

    printf("\n2. Scope-Based Timing (RAII):\n");
    {
        auto timer = profiler.time_scope("scoped_operation");

        // Complex operation
        cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        complex_math_kernel<<<(N+255)/256, 256>>>(d_input, d_temp, N);
        simple_math_kernel<<<(N+255)/256, 256>>>(d_temp, d_output, N);
        cudaMemcpy(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Timer automatically stops when scope ends
    }

    printf("\n3. Benchmarking Operations:\n");

    // Benchmark different kernel configurations
    profiler.benchmark_operation("kernel_256_threads", [&]() {
        complex_math_kernel<<<(N+255)/256, 256>>>(d_input, d_output, N);
        cudaDeviceSynchronize();
    }, 5);

    profiler.benchmark_operation("kernel_512_threads", [&]() {
        complex_math_kernel<<<(N+511)/512, 512>>>(d_input, d_output, N);
        cudaDeviceSynchronize();
    }, 5);

    profiler.benchmark_operation("kernel_1024_threads", [&]() {
        complex_math_kernel<<<(N+1023)/1024, 1024>>>(d_input, d_output, N);
        cudaDeviceSynchronize();
    }, 5);

    printf("\n4. Pipeline Timing:\n");

    // Time overlapped operations
    profiler.start_timing("overlapped_pipeline");

    cudaMemcpyAsync(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream1);
    complex_math_kernel<<<(N+255)/256, 256, 0, stream1>>>(d_input, d_temp, N);
    simple_math_kernel<<<(N+255)/256, 256, 0, stream2>>>(d_temp, d_output, N);
    cudaMemcpyAsync(h_data, d_output, N * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    profiler.stop_timing("overlapped_pipeline");

    // Print comprehensive results
    profiler.print_all_stats();

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
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
```

## Advanced Synchronization Patterns

Events enable sophisticated synchronization patterns beyond basic stream coordination, including complex dependency graphs and multi-stage pipeline coordination.

### Event-Based Coordination Patterns

```cpp
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

// Demonstrate complex event-driven coordination
void demonstrate_event_coordination() {
    printf("=== Event-Driven Coordination Demo ===\n");

    // Create streams
    std::vector<cudaStream_t> streams(4);
    for (auto& stream : streams) {
        cudaStreamCreate(&stream);
    }

    EventCoordinator coordinator(streams);

    // Create test data
    const int N = 1024 * 1024;
    float *d_data1, *d_data2, *d_data3, *d_data4, *d_temp;

    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMalloc(&d_data3, N * sizeof(float));
    cudaMalloc(&d_data4, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));

    // Define work functions
    auto init_data = [=](cudaStream_t stream) {
        cudaMemsetAsync(d_data1, 0, N * sizeof(float), stream);
        initialization_kernel<<<(N+255)/256, 256, 0, stream>>>(d_data1, N);
    };

    auto process_stage1 = [=](cudaStream_t stream) {
        complex_math_kernel<<<(N+255)/256, 256, 0, stream>>>(d_data1, d_data2, N);
    };

    auto process_stage2a = [=](cudaStream_t stream) {
        simple_math_kernel<<<(N+255)/256, 256, 0, stream>>>(d_data2, d_data3, N);
    };

    auto process_stage2b = [=](cudaStream_t stream) {
        complex_math_kernel<<<(N+255)/256, 256, 0, stream>>>(d_data2, d_temp, N);
    };

    auto final_combine = [=](cudaStream_t stream) {
        combine_kernel<<<(N+255)/256, 256, 0, stream>>>(d_data3, d_temp, d_data4, N);
    };

    // Build dependency graph
    // Stage 0: Initialize data
    coordinator.add_node("initialize", {}, init_data);

    // Stage 1: Process initial data (depends on initialize)
    coordinator.add_node("stage1", {"initialize"}, process_stage1);

    // Stage 2: Parallel processing (both depend on stage1)
    coordinator.add_node("stage2a", {"stage1"}, process_stage2a);
    coordinator.add_node("stage2b", {"stage1"}, process_stage2b);

    // Stage 3: Combine results (depends on both stage2a and stage2b)
    coordinator.add_node("combine", {"stage2a", "stage2b"}, final_combine);

    // Print graph structure
    coordinator.print_graph_structure();

    // Execute the graph
    auto start_time = std::chrono::high_resolution_clock::now();
    coordinator.execute_graph();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("Total graph execution time: %ld ms\n", execution_time.count());

    // Cleanup
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }

    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
    cudaFree(d_data4);
    cudaFree(d_temp);
}

__global__ void initialization_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = tid * 0.001f;
    }
}

__global__ void combine_kernel(float* data1, float* data2, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = (data1[tid] + data2[tid]) * 0.5f;
    }
}
```
