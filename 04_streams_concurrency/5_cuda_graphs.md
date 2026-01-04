# CUDA Graphs Deep Dive

CUDA Graphs represent a paradigm shift from dynamic kernel launches to static execution graphs, enabling dramatic performance improvements for repetitive workloads by reducing launch overhead and enabling advanced optimizations.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Related: [Stream Fundamentals](1_stream_fundamentals.md)**

---

## Graph Fundamentals and Architecture

CUDA Graphs capture sequences of GPU operations into a static directed acyclic graph (DAG), allowing the CUDA runtime to optimize execution and minimize overhead.

### Comprehensive Graph Management System

**Source Code**: [`../src/04_streams_concurrency/graph_manager.h`](../src/04_streams_concurrency/graph_manager.h)

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

## Advanced Graph Patterns and Optimization

### Dynamic Graph Updates and Parameter Modification

```cpp
// Full implementation available in ../src/04_streams_concurrency/advanced_graph_patterns.cuh
#include "../src/04_streams_concurrency/advanced_graph_patterns.cuh"

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

## Production Graph Optimization Strategies

### Enterprise-Grade Graph Management

```cpp
// Full implementation available in ../src/04_streams_concurrency/production_graph_optimizer.cuh
#include "../src/04_streams_concurrency/production_graph_optimizer.cuh"

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
