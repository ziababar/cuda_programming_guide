#  CUDA Streams & Concurrency Complete Guide

CUDA streams are the backbone of high-performance GPU programming, enabling asynchronous execution, memory transfer overlap, and sophisticated pipeline orchestration. Understanding streams deeply is essential for achieving optimal GPU utilization and building scalable parallel applications.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Related: [Memory Hierarchy](../02_memory_hierarchy/1_cuda_memory_hierarchy.md)** | **Architecture: [Execution Model](../01_execution_model/1_cuda_execution_model.md)**

---

##  **Table of Contents**

1. [ Stream Fundamentals](#-stream-fundamentals)
2. [ Asynchronous Operations](#-asynchronous-operations)
3. [ Memory Transfer Optimization](#-memory-transfer-optimization)
4. [ Event-Driven Programming](#-event-driven-programming)
5. [ CUDA Graphs Deep Dive](#-cuda-graphs-deep-dive)
6. [ Advanced Stream Patterns](#-advanced-stream-patterns)
7. [ Multi-GPU Coordination](#-multi-gpu-coordination)
8. [ Performance Analysis](#-performance-analysis)
9. [ Debugging and Troubleshooting](#-debugging-and-troubleshooting)
10. [ Production Patterns](#-production-patterns)

---

##  **Quick Reference**

### **Stream Hierarchy:**
```
Host Application
 Default Stream (Blocking)
 Explicit Streams (Async)
    Memory Transfers
    Kernel Executions
    Event Synchronization
 CUDA Graphs (Static DAG)
     Node Dependencies
     Optimized Execution
```

### **Key Performance Concepts:**
| Concept | Description | Performance Impact |
|---------|-------------|-------------------|
| **Stream Overlap** | Concurrent compute + memory transfer | 2-4x throughput improvement |
| **Pinned Memory** | Host memory accessible by DMA | 2-3x transfer speed |
| **Event Synchronization** | Fine-grained stream coordination | Minimal overhead |
| **CUDA Graphs** | Static execution DAG | 50-90% launch overhead reduction |
| **Multi-Stream** | Parallel execution contexts | Near-linear scaling |

---

##  **Stream Fundamentals**

CUDA streams represent ordered sequences of GPU operations that execute asynchronously with respect to the host and other streams, enabling sophisticated concurrency patterns.

###  **Stream Types and Properties**

#### **Stream Hierarchy and Characteristics:**
```cpp
#include <cstdio>
#include <cmath>
#include <chrono>

// Forward declaration of kernel
__global__ void simple_kernel(float* data, int N);

// Comprehensive stream type demonstration
void demonstrate_stream_fundamentals() {
    printf("=== CUDA Stream Fundamentals ===\n");

    // 1. Default Stream (Stream 0) - Synchronous Behavior
    printf("1. Default Stream Characteristics:\n");
    printf("   - Synchronous with host\n");
    printf("   - Blocks other streams until completion\n");
    printf("   - Used when no explicit stream specified\n\n");

    float *d_data1, *d_data2;
    size_t size = 1024 * sizeof(float);

    cudaMalloc(&d_data1, size);
    cudaMalloc(&d_data2, size);

    // Default stream operations execute sequentially
    auto start = std::chrono::high_resolution_clock::now();

    cudaMemset(d_data1, 0, size);                    // Blocks host
    simple_kernel<<<256, 256>>>(d_data1, 1024);     // Blocks until memset done
    cudaMemset(d_data2, 1, size);                    // Blocks until kernel done
    cudaDeviceSynchronize();                         // Wait for completion

    auto end = std::chrono::high_resolution_clock::now();
    auto sequential_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("   Sequential execution time: %ld μs\n\n", sequential_time.count());

    // 2. Explicit Streams - Asynchronous Behavior
    printf("2. Explicit Stream Characteristics:\n");
    printf("   - Asynchronous with host\n");
    printf("   - Can execute concurrently with other streams\n");
    printf("   - Enable overlap and pipelining\n");

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    start = std::chrono::high_resolution_clock::now();

    // These can execute concurrently
    cudaMemsetAsync(d_data1, 0, size, stream1);
    cudaMemsetAsync(d_data2, 1, size, stream2);
    simple_kernel<<<256, 256, 0, stream1>>>(d_data1, 1024);
    simple_kernel<<<256, 256, 0, stream2>>>(d_data2, 1024);

    // Synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    end = std::chrono::high_resolution_clock::now();
    auto concurrent_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("   Concurrent execution time: %ld μs\n", concurrent_time.count());
    printf("   Speedup: %.2fx\n\n", (float)sequential_time.count() / concurrent_time.count());

    // 3. Stream Properties and Configuration
    printf("3. Stream Properties:\n");

    // Query stream priorities
    int low_priority, high_priority;
    cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);
    printf("   Priority range: %d (low) to %d (high)\n", low_priority, high_priority);

    // Create priority streams
    cudaStream_t high_prio_stream, low_prio_stream;
    cudaStreamCreateWithPriority(&high_prio_stream, cudaStreamNonBlocking, high_priority);
    cudaStreamCreateWithPriority(&low_prio_stream, cudaStreamNonBlocking, low_priority);

    printf("   High priority stream created\n");
    printf("   Low priority stream created\n");

    // Test non-blocking vs blocking behavior
    cudaStream_t blocking_stream, non_blocking_stream;
    cudaStreamCreateWithFlags(&blocking_stream, cudaStreamDefault);        // Blocking
    cudaStreamCreateWithFlags(&non_blocking_stream, cudaStreamNonBlocking); // Non-blocking

    printf("   Blocking stream: synchronizes with default stream\n");
    printf("   Non-blocking stream: independent execution\n\n");

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(high_prio_stream);
    cudaStreamDestroy(low_prio_stream);
    cudaStreamDestroy(blocking_stream);
    cudaStreamDestroy(non_blocking_stream);
    cudaFree(d_data1);
    cudaFree(d_data2);
}

__global__ void simple_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Simple computation to demonstrate stream behavior
        data[tid] = tid * 2.0f + sin(tid * 0.01f);
    }
}
```

#### **Stream Execution Model:**
```cpp
#include <cstdio>
#include <cmath>

// Forward declarations of kernels
__global__ void preprocessing_kernel(float* data, int N);
__global__ void processing_kernel(float* data, int N);
__global__ void compute_intensive_kernel(float* data, int N);
__global__ void postprocessing_kernel(float* data, int N);
__global__ void initialization_kernel(float* data, int N);

// Demonstrate FIFO ordering and inter-stream concurrency
void demonstrate_stream_execution_model() {
    printf("=== Stream Execution Model ===\n");

    const int N = 1024 * 1024;
    float *h_data, *d_data1, *d_data2, *d_data3;

    // Allocate pinned memory for optimal transfer performance
    cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMalloc(&d_data3, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.001f;
    }

    // Create streams with different characteristics
    cudaStream_t stream_a, stream_b, stream_c;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);
    cudaStreamCreate(&stream_c);

    printf("1. FIFO Ordering Within Streams:\n");
    printf("   Operations within each stream execute in submission order\n");

    // Stream A: Sequential pipeline
    cudaMemcpyAsync(d_data1, h_data, N * sizeof(float),
                   cudaMemcpyHostToDevice, stream_a);         // Order: 1
    preprocessing_kernel<<<(N+255)/256, 256, 0, stream_a>>>(d_data1, N);    // Order: 2
    processing_kernel<<<(N+255)/256, 256, 0, stream_a>>>(d_data1, N);       // Order: 3
    cudaMemcpyAsync(h_data, d_data1, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_a);         // Order: 4

    printf("   Stream A: H2D -> Preprocess -> Process -> D2H\n");

    // Stream B: Different pipeline
    cudaMemcpyAsync(d_data2, h_data, N * sizeof(float),
                   cudaMemcpyHostToDevice, stream_b);         // Concurrent with Stream A
    compute_intensive_kernel<<<(N+255)/256, 256, 0, stream_b>>>(d_data2, N); // Different work
    postprocessing_kernel<<<(N+255)/256, 256, 0, stream_b>>>(d_data2, N);

    printf("   Stream B: H2D -> Intensive -> Postprocess (concurrent)\n");

    // Stream C: Memory operations
    cudaMemsetAsync(d_data3, 0, N * sizeof(float), stream_c);              // Concurrent init
    initialization_kernel<<<(N+255)/256, 256, 0, stream_c>>>(d_data3, N);

    printf("   Stream C: Memset -> Initialize (concurrent)\n\n");

    printf("2. Inter-Stream Concurrency:\n");
    printf("   Different streams can execute concurrently\n");
    printf("   GPU scheduler interleaves stream operations\n");
    printf("   Actual concurrency depends on resource availability\n\n");

    // Wait for all streams to complete
    cudaStreamSynchronize(stream_a);
    cudaStreamSynchronize(stream_b);
    cudaStreamSynchronize(stream_c);

    printf("3. Synchronization Points:\n");
    printf("   cudaStreamSynchronize() - Wait for specific stream\n");
    printf("   cudaDeviceSynchronize() - Wait for all streams\n");
    printf("   Events - Fine-grained inter-stream dependencies\n");

    // Cleanup
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    cudaStreamDestroy(stream_c);
    cudaFreeHost(h_data);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
}

__global__ void preprocessing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = sqrt(fabs(data[tid]));
    }
}

__global__ void processing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = sin(data[tid]) + cos(data[tid]);
    }
}

__global__ void compute_intensive_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float value = data[tid];
        // Intensive computation
        for (int i = 0; i < 100; i++) {
            value = sin(value) + cos(value);
        }
        data[tid] = value;
    }
}

__global__ void postprocessing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

__global__ void initialization_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = tid * 0.01f;
    }
}
```

###  **Stream Management Patterns**

#### **Advanced Stream Management:**
```cpp
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <iostream>

// Sophisticated stream management for production applications
class StreamManager {
private:
    std::vector<cudaStream_t> streams;
    std::vector<int> stream_priorities;
    std::vector<bool> stream_busy;
    std::vector<std::string> stream_tags;
    int num_streams;
    int current_stream_index;

public:
    StreamManager(int count, bool use_priorities = false,
                 const std::vector<std::string>& tags = {})
        : num_streams(count), current_stream_index(0) {

        streams.resize(count);
        stream_priorities.resize(count);
        stream_busy.resize(count, false);
        stream_tags.resize(count);

        // Set up stream tags
        for (int i = 0; i < count; i++) {
            if (i < tags.size()) {
                stream_tags[i] = tags[i];
            } else {
                stream_tags[i] = "Stream_" + std::to_string(i);
            }
        }

        if (use_priorities) {
            int low_priority, high_priority;
            cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);

            // Distribute priorities evenly across streams
            for (int i = 0; i < count; i++) {
                int priority = high_priority +
                              (i * (low_priority - high_priority)) / std::max(1, count - 1);
                stream_priorities[i] = priority;

                cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, priority);
                printf("Created %s with priority %d\n", stream_tags[i].c_str(), priority);
            }
        } else {
            // Standard streams with no explicit priorities
            for (int i = 0; i < count; i++) {
                cudaStreamCreate(&streams[i]);
                stream_priorities[i] = 0;
                printf("Created %s (standard priority)\n", stream_tags[i].c_str());
            }
        }

        printf("StreamManager initialized with %d streams\n", count);
    }

    // Get stream by index
    cudaStream_t get_stream(int index) const {
        if (index >= 0 && index < num_streams) {
            return streams[index];
        }
        return streams[0];  // Fallback to first stream
    }

    // Round-robin stream allocation
    cudaStream_t get_next_stream() {
        cudaStream_t stream = streams[current_stream_index];
        current_stream_index = (current_stream_index + 1) % num_streams;
        return stream;
    }

    // Get least busy stream
    cudaStream_t get_available_stream() {
        // First, try to find a completely idle stream
        for (int i = 0; i < num_streams; i++) {
            cudaError_t status = cudaStreamQuery(streams[i]);
            if (status == cudaSuccess) {  // Stream is idle
                stream_busy[i] = false;
                return streams[i];
            }
        }

        // If all streams are busy, return the next one in round-robin
        return get_next_stream();
    }

    // Get stream optimized for specific workload
    cudaStream_t get_stream_for_workload(const std::string& workload_type) {
        if (workload_type == "high_priority") {
            // Return highest priority stream
            int best_stream = 0;
            int best_priority = stream_priorities[0];

            for (int i = 1; i < num_streams; i++) {
                if (stream_priorities[i] < best_priority) {  // Lower number = higher priority
                    best_priority = stream_priorities[i];
                    best_stream = i;
                }
            }

            printf("Assigned %s for high priority workload\n", stream_tags[best_stream].c_str());
            return streams[best_stream];

        } else if (workload_type == "memory_intensive") {
            // For memory-intensive work, prefer available streams to avoid competition
            return get_available_stream();

        } else if (workload_type == "compute_intensive") {
            // For compute-intensive work, any stream is fine
            return get_next_stream();

        } else {
            // Default case
            return get_available_stream();
        }
    }

    // Synchronize specific stream
    void synchronize_stream(int index) {
        if (index >= 0 && index < num_streams) {
            cudaStreamSynchronize(streams[index]);
            stream_busy[index] = false;
            printf("%s synchronized\n", stream_tags[index].c_str());
        }
    }

    // Synchronize all streams
    void synchronize_all() {
        printf("Synchronizing all streams...\n");
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
            stream_busy[i] = false;
        }
        printf("All streams synchronized\n");
    }

    // Query status of all streams
    void print_status() {
        printf("=== Stream Status ===\n");
        for (int i = 0; i < num_streams; i++) {
            cudaError_t status = cudaStreamQuery(streams[i]);
            const char* status_str = (status == cudaSuccess) ? "Idle" : "Busy";
            printf("  %s (priority %d): %s\n",
                   stream_tags[i].c_str(), stream_priorities[i], status_str);
        }
        printf("====================\n");
    }

    // Get performance statistics
    void print_performance_stats() {
        printf("=== Stream Performance Stats ===\n");
        // This would require more sophisticated tracking in a real implementation
        printf("Total streams: %d\n", num_streams);
        printf("Priority range: %d to %d\n",
               *std::min_element(stream_priorities.begin(), stream_priorities.end()),
               *std::max_element(stream_priorities.begin(), stream_priorities.end()));
        printf("Current allocation index: %d\n", current_stream_index);
        printf("===============================\n");
    }

    ~StreamManager() {
        printf("Destroying StreamManager...\n");
        synchronize_all();

        for (int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(streams[i]);
            printf("Destroyed %s\n", stream_tags[i].c_str());
        }

        printf("StreamManager cleanup complete\n");
    }
};

// Usage example
void demonstrate_stream_manager() {
    printf("=== Stream Manager Demo ===\n");

    // Create manager with priority streams and custom tags
    std::vector<std::string> tags = {"MemoryOps", "Compute", "HighPrio", "Background"};
    StreamManager manager(4, true, tags);

    // Simulate different workload assignments
    const int N = 1024 * 1024;
    float *d_data1, *d_data2, *d_data3, *d_data4;

    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMalloc(&d_data3, N * sizeof(float));
    cudaMalloc(&d_data4, N * sizeof(float));

    // Assign workloads to appropriate streams
    cudaStream_t memory_stream = manager.get_stream_for_workload("memory_intensive");
    cudaStream_t compute_stream = manager.get_stream_for_workload("compute_intensive");
    cudaStream_t priority_stream = manager.get_stream_for_workload("high_priority");
    cudaStream_t background_stream = manager.get_next_stream();

    // Launch operations
    cudaMemsetAsync(d_data1, 0, N * sizeof(float), memory_stream);
    compute_intensive_kernel<<<(N+255)/256, 256, 0, compute_stream>>>(d_data2, N);
    preprocessing_kernel<<<(N+255)/256, 256, 0, priority_stream>>>(d_data3, N);
    processing_kernel<<<(N+255)/256, 256, 0, background_stream>>>(d_data4, N);

    // Check status
    manager.print_status();

    // Synchronize and get final status
    manager.synchronize_all();
    manager.print_performance_stats();

    // Cleanup
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
    cudaFree(d_data4);
}
```

---

##  **Asynchronous Operations**

Asynchronous execution is the cornerstone of high-performance GPU programming, enabling overlapped computation, memory transfer concurrency, and sophisticated pipeline orchestration.

###  **Compute-Transfer Overlap**

The ability to overlap computation with memory transfers is one of the most powerful features of CUDA streams, often yielding 2-4x throughput improvements.

#### **Basic Overlap Patterns:**
```cpp
#include <cstdio>
#include <cmath>
#include <vector>

// Forward declaration
__global__ void complex_processing_kernel(float* input, float* output, int N);
void pipeline_processing_demo(float* h_input, float* h_output, float* d_input, float* d_output, int N, cudaStream_t* streams, int num_streams);

// Comprehensive compute-transfer overlap demonstration
void demonstrate_compute_transfer_overlap() {
    const int N = 4 * 1024 * 1024;  // 4M elements
    const int chunk_size = N / 4;    // Process in 4 chunks

    float *h_input, *h_output, *d_input, *d_output;

    // Allocate pinned memory for maximum transfer speed
    cudaHostAlloc(&h_input, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_output, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = sin(i * 0.01f);
    }

    // Create multiple streams for overlap
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Timing for comparison
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("=== Compute-Transfer Overlap Analysis ===\n");

    // Method 1: Sequential (no overlap)
    printf("1. Sequential Processing (No Overlap):\n");
    cudaEventRecord(start);

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    complex_processing_kernel<<<(N+255)/256, 256>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float sequential_time;
    cudaEventElapsedTime(&sequential_time, start, stop);
    printf("   Sequential time: %.2f ms\n", sequential_time);

    // Method 2: Chunked with overlap
    printf("2. Chunked Processing (With Overlap):\n");
    cudaEventRecord(start);

    for (int chunk = 0; chunk < num_streams; chunk++) {
        int offset = chunk * chunk_size;
        cudaStream_t stream = streams[chunk];

        // Async copy input chunk
        cudaMemcpyAsync(&d_input[offset], &h_input[offset],
                       chunk_size * sizeof(float),
                       cudaMemcpyHostToDevice, stream);

        // Process chunk
        complex_processing_kernel<<<(chunk_size+255)/256, 256, 0, stream>>>(
            &d_input[offset], &d_output[offset], chunk_size);

        // Async copy output chunk
        cudaMemcpyAsync(&h_output[offset], &d_output[offset],
                       chunk_size * sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
    }

    // Wait for all streams to complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float overlapped_time;
    cudaEventElapsedTime(&overlapped_time, start, stop);
    printf("   Overlapped time: %.2f ms\n", overlapped_time);
    printf("   Speedup: %.2fx\n", sequential_time / overlapped_time);

    // Method 3: Advanced pipeline processing
    printf("3. Pipeline Processing (Advanced Overlap):\n");
    pipeline_processing_demo(h_input, h_output, d_input, d_output, N, streams, num_streams);

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void complex_processing_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = input[tid];

        // Complex computation to make overlap beneficial
        for (int i = 0; i < 100; i++) {
            value = sin(value) + cos(value);
            value = sqrt(fabs(value) + 1.0f);
            value = log(value + 1.0f);
        }

        output[tid] = value;
    }
}
```

#### **Advanced Pipeline Processing:**
```cpp
#include <cstdio>
#include <cmath>
#include <vector>

// Forward declarations
__global__ void stage1_kernel(float* input, float* output, int N);
__global__ void stage2_kernel(float* input, float* output, int N);

// Sophisticated pipeline with multiple processing stages
void pipeline_processing_demo(float* h_input, float* h_output,
                             float* d_input, float* d_output,
                             int N, cudaStream_t* streams, int num_streams) {

    const int chunk_size = N / (num_streams * 2);  // Smaller chunks for better overlap
    float *d_temp;
    cudaMalloc(&d_temp, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("   Pipeline Processing:\n");
    cudaEventRecord(start);

    // Create events for pipeline synchronization
    std::vector<cudaEvent_t> stage_events(num_streams * 3);
    for (auto& event : stage_events) {
        cudaEventCreate(&event);
    }

    int chunks_processed = 0;
    int total_chunks = N / chunk_size;

    // Pipeline: Input Transfer -> Stage1 -> Stage2 -> Output Transfer
    for (int chunk = 0; chunk < total_chunks; chunk++) {
        int stream_id = chunk % num_streams;
        cudaStream_t stream = streams[stream_id];
        int offset = chunk * chunk_size;

        // Stage 1: Input transfer
        cudaMemcpyAsync(&d_input[offset], &h_input[offset],
                       chunk_size * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        cudaEventRecord(stage_events[stream_id * 3 + 0], stream);

        // Stage 2: First processing phase
        stage1_kernel<<<(chunk_size+255)/256, 256, 0, stream>>>(
            &d_input[offset], &d_temp[offset], chunk_size);
        cudaEventRecord(stage_events[stream_id * 3 + 1], stream);

        // Stage 3: Second processing phase
        stage2_kernel<<<(chunk_size+255)/256, 256, 0, stream>>>(
            &d_temp[offset], &d_output[offset], chunk_size);
        cudaEventRecord(stage_events[stream_id * 3 + 2], stream);

        // Stage 4: Output transfer
        cudaMemcpyAsync(&h_output[offset], &d_output[offset],
                       chunk_size * sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float pipeline_time;
    cudaEventElapsedTime(&pipeline_time, start, stop);
    printf("      Pipeline time: %.2f ms\n", pipeline_time);

    // Cleanup
    for (auto& event : stage_events) {
        cudaEventDestroy(event);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_temp);
}

__global__ void stage1_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = sqrt(fabs(input[tid]));
    }
}

__global__ void stage2_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = sin(input[tid]) + cos(input[tid]);
    }
}
```

###  **Stream Synchronization Mechanisms**

#### **Comprehensive Synchronization Patterns:**
```cpp
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <cmath>

// Forward declarations
__global__ void parallel_work_kernel(int stream_id);
__global__ void join_work_kernel();
__global__ void pipeline_stage_kernel(int stage, int iteration);
__global__ void initial_processing_kernel();
__global__ void parallel_processing_kernel(int branch_id);
__global__ void aggregation_kernel();

// Advanced synchronization techniques for complex workflows
class StreamSynchronizer {
private:
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> sync_events;
    std::map<std::string, int> named_streams;

public:
    StreamSynchronizer(int num_streams) {
        streams.resize(num_streams);
        sync_events.resize(num_streams);

        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&sync_events[i]);

            // Create named streams for easier reference
            std::string name = "stream_" + std::to_string(i);
            named_streams[name] = i;
        }

        printf("StreamSynchronizer created with %d streams\n", num_streams);
    }

    // Get stream by index or name
    cudaStream_t get_stream(int index) const {
        return streams[index % streams.size()];
    }

    cudaStream_t get_stream(const std::string& name) const {
        auto it = named_streams.find(name);
        if (it != named_streams.end()) {
            return streams[it->second];
        }
        return streams[0];  // Default fallback
    }

    // Barrier synchronization across all streams
    void barrier_sync() {
        printf("Performing barrier synchronization...\n");

        // Record events in all streams
        for (int i = 0; i < streams.size(); i++) {
            cudaEventRecord(sync_events[i], streams[i]);
        }

        // Wait for all events in all streams
        for (int i = 0; i < streams.size(); i++) {
            for (int j = 0; j < streams.size(); j++) {
                if (i != j) {
                    cudaStreamWaitEvent(streams[i], sync_events[j], 0);
                }
            }
        }

        printf("Barrier synchronization complete\n");
    }

    // Producer-consumer synchronization
    void producer_consumer_sync(int producer_stream, int consumer_stream,
                               cudaEvent_t& sync_event) {
        // Producer signals completion
        cudaEventRecord(sync_event, streams[producer_stream]);

        // Consumer waits for producer
        cudaStreamWaitEvent(streams[consumer_stream], sync_event, 0);

        printf("Producer-consumer sync: stream %d -> stream %d\n",
               producer_stream, consumer_stream);
    }

    // Fork-join pattern
    void fork_join_pattern(const std::vector<int>& parallel_streams,
                          int join_stream) {
        printf("Executing fork-join pattern...\n");

        // Fork: Launch work on parallel streams
        for (int stream_id : parallel_streams) {
            parallel_work_kernel<<<256, 256, 0, streams[stream_id]>>>(stream_id);
            cudaEventRecord(sync_events[stream_id], streams[stream_id]);
        }

        // Join: Wait for all parallel work to complete
        for (int stream_id : parallel_streams) {
            cudaStreamWaitEvent(streams[join_stream], sync_events[stream_id], 0);
        }

        // Continue with joined work
        join_work_kernel<<<256, 256, 0, streams[join_stream]>>>();

        printf("Fork-join pattern complete\n");
    }

    // Pipeline stage synchronization
    void pipeline_stage_sync(int stage_count, int iterations) {
        printf("Executing %d-stage pipeline for %d iterations...\n",
               stage_count, iterations);

        for (int iter = 0; iter < iterations; iter++) {
            for (int stage = 0; stage < stage_count; stage++) {
                int stream_id = stage % streams.size();

                // Wait for previous stage if not first stage
                if (stage > 0) {
                    int prev_stream = (stage - 1) % streams.size();
                    cudaStreamWaitEvent(streams[stream_id], sync_events[prev_stream], 0);
                }

                // Execute stage
                pipeline_stage_kernel<<<128, 128, 0, streams[stream_id]>>>(stage, iter);

                // Signal stage completion
                cudaEventRecord(sync_events[stream_id], streams[stream_id]);
            }
        }

        printf("Pipeline execution complete\n");
    }

    // Advanced dependency graph execution
    void execute_dependency_graph() {
        printf("Executing complex dependency graph...\n");

        // Example dependency graph:
        // Stream 0: Initial data processing
        // Stream 1 & 2: Parallel processing (depend on stream 0)
        // Stream 3: Final aggregation (depends on streams 1 & 2)

        // Stage 1: Initial processing
        initial_processing_kernel<<<256, 256, 0, streams[0]>>>();
        cudaEventRecord(sync_events[0], streams[0]);

        // Stage 2: Parallel processing (both depend on stage 1)
        cudaStreamWaitEvent(streams[1], sync_events[0], 0);
        cudaStreamWaitEvent(streams[2], sync_events[0], 0);

        parallel_processing_kernel<<<256, 256, 0, streams[1]>>>(1);
        parallel_processing_kernel<<<256, 256, 0, streams[2]>>>(2);

        cudaEventRecord(sync_events[1], streams[1]);
        cudaEventRecord(sync_events[2], streams[2]);

        // Stage 3: Final aggregation (depends on both parallel stages)
        cudaStreamWaitEvent(streams[3], sync_events[1], 0);
        cudaStreamWaitEvent(streams[3], sync_events[2], 0);

        aggregation_kernel<<<256, 256, 0, streams[3]>>>();

        printf("Dependency graph execution complete\n");
    }

    // Synchronize all streams
    void synchronize_all() {
        for (auto& stream : streams) {
            cudaStreamSynchronize(stream);
        }
    }

    ~StreamSynchronizer() {
        synchronize_all();

        for (auto& stream : streams) {
            cudaStreamDestroy(stream);
        }
        for (auto& event : sync_events) {
            cudaEventDestroy(event);
        }

        printf("StreamSynchronizer destroyed\n");
    }
};

// Kernel implementations for synchronization demo
__global__ void parallel_work_kernel(int stream_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Simulate different amounts of work per stream
    for (int i = 0; i < (stream_id + 1) * 100; i++) {
        float dummy = sin(tid * 0.01f + i);
    }

    if (tid == 0) {
        printf("Stream %d parallel work complete\n", stream_id);
    }
}

__global__ void join_work_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Work that depends on all parallel streams completing
    float result = cos(tid * 0.01f);

    if (tid == 0) {
        printf("Join work complete\n");
    }
}

__global__ void pipeline_stage_kernel(int stage, int iteration) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Stage-specific processing
    float result = sin(tid * stage * iteration * 0.001f);

    if (tid == 0) {
        printf("Stage %d, iteration %d complete\n", stage, iteration);
    }
}

__global__ void initial_processing_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initial data processing
    float result = tid * 0.01f;

    if (tid == 0) {
        printf("Initial processing complete\n");
    }
}

__global__ void parallel_processing_kernel(int branch_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Branch-specific processing
    float result = sin(tid * branch_id * 0.01f);

    if (tid == 0) {
        printf("Parallel processing branch %d complete\n", branch_id);
    }
}

__global__ void aggregation_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Final aggregation
    float result = cos(tid * 0.01f);

    if (tid == 0) {
        printf("Aggregation complete\n");
    }
}
```

###  **Dynamic Stream Management**

#### **Adaptive Stream Allocation:**
```cpp
#include <vector>
#include <queue>
#include <map>
#include <chrono>
#include <cstdio>

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
```

---

##  **Memory Transfer Optimization**

Memory transfer optimization is crucial for achieving peak performance in CUDA applications. Understanding the memory hierarchy, transfer patterns, and bandwidth utilization strategies can significantly impact overall application throughput.

###  **Pinned Memory Deep Dive**

Pinned (page-locked) memory is essential for achieving maximum memory transfer bandwidth and enabling true asynchronous operations.

#### **Comprehensive Pinned Memory Management:**
```cpp
#include <map>
#include <cstdio>
#include <vector>
#include <cmath>

// Forward declaration
__global__ void zero_copy_kernel(float* data, int N);

// Advanced pinned memory allocation and management
class PinnedMemoryManager {
private:
    std::map<void*, size_t> allocated_blocks;
    std::map<void*, cudaHostAllocFlags> allocation_flags;
    size_t total_allocated;
    size_t max_allocation_limit;

public:
    PinnedMemoryManager(size_t max_limit = 2ULL * 1024 * 1024 * 1024) // 2GB default
        : total_allocated(0), max_allocation_limit(max_limit) {
        printf("PinnedMemoryManager initialized (max: %.2f GB)\n",
               max_limit / (1024.0 * 1024.0 * 1024.0));
    }

    // Allocate pinned memory with various flags
    void* allocate(size_t size, cudaHostAllocFlags flags = cudaHostAllocDefault) {
        if (total_allocated + size > max_allocation_limit) {
            printf("Warning: Allocation would exceed limit (%.2f GB used of %.2f GB)\n",
                   total_allocated / (1024.0 * 1024.0 * 1024.0),
                   max_allocation_limit / (1024.0 * 1024.0 * 1024.0));
            return nullptr;
        }

        void* ptr = nullptr;
        cudaError_t result = cudaHostAlloc(&ptr, size, flags);

        if (result == cudaSuccess && ptr != nullptr) {
            allocated_blocks[ptr] = size;
            allocation_flags[ptr] = flags;
            total_allocated += size;

            printf("Allocated %.2f MB pinned memory (flags: %d)\n",
                   size / (1024.0 * 1024.0), flags);

            return ptr;
        } else {
            printf("Failed to allocate pinned memory: %s\n", cudaGetErrorString(result));
            return nullptr;
        }
    }

    // Free pinned memory
    void deallocate(void* ptr) {
        auto it = allocated_blocks.find(ptr);
        if (it != allocated_blocks.end()) {
            size_t size = it->second;
            total_allocated -= size;

            cudaFreeHost(ptr);
            allocated_blocks.erase(it);
            allocation_flags.erase(ptr);

            printf("Freed %.2f MB pinned memory\n", size / (1024.0 * 1024.0));
        }
    }

    // Get memory statistics
    void print_statistics() {
        printf("=== Pinned Memory Statistics ===\n");
        printf("Total allocated: %.2f MB\n", total_allocated / (1024.0 * 1024.0));
        printf("Number of blocks: %zu\n", allocated_blocks.size());
        printf("Utilization: %.1f%%\n",
               (total_allocated * 100.0) / max_allocation_limit);

        // Break down by allocation flags
        std::map<cudaHostAllocFlags, size_t> flag_usage;
        for (const auto& pair : allocation_flags) {
            flag_usage[pair.second] += allocated_blocks[pair.first];
        }

        for (const auto& pair : flag_usage) {
            printf("Flag %d usage: %.2f MB\n",
                   pair.first, pair.second / (1024.0 * 1024.0));
        }
        printf("===============================\n");
    }

    ~PinnedMemoryManager() {
        // Free all remaining allocations
        for (auto& pair : allocated_blocks) {
            cudaFreeHost(pair.first);
        }
        printf("PinnedMemoryManager cleanup complete (freed %.2f MB)\n",
               total_allocated / (1024.0 * 1024.0));
    }
};

// Demonstrate different pinned memory allocation types
void demonstrate_pinned_memory_types() {
    printf("=== Pinned Memory Types Comparison ===\n");

    const size_t test_size = 64 * 1024 * 1024; // 64MB
    PinnedMemoryManager manager;

    // Test different allocation flags
    struct TestConfig {
        cudaHostAllocFlags flags;
        const char* description;
    } configs[] = {
        {cudaHostAllocDefault, "Default pinned memory"},
        {cudaHostAllocWriteCombined, "Write-combined (faster H2D)"},
        {cudaHostAllocMapped, "Mapped (zero-copy access)"},
        {cudaHostAllocPortable, "Portable across contexts"},
        {cudaHostAllocWriteCombined | cudaHostAllocMapped, "Write-combined + Mapped"}
    };

    std::vector<void*> test_buffers;
    float *d_buffer;
    cudaMalloc(&d_buffer, test_size);

    // Create streams for async operations
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Test each configuration
    for (const auto& config : configs) {
        printf("\nTesting: %s\n", config.description);

        void* h_buffer = manager.allocate(test_size, config.flags);
        if (!h_buffer) continue;

        test_buffers.push_back(h_buffer);

        // Initialize data
        for (int i = 0; i < test_size / sizeof(float); i++) {
            ((float*)h_buffer)[i] = i * 0.001f;
        }

        // Measure transfer performance
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Host to Device transfer
        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++) {
            cudaMemcpyAsync(d_buffer, h_buffer, test_size,
                          cudaMemcpyHostToDevice, stream1);
        }
        cudaStreamSynchronize(stream1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float h2d_time;
        cudaEventElapsedTime(&h2d_time, start, stop);
        float h2d_bandwidth = (test_size * 10) / (h2d_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        // Device to Host transfer
        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++) {
            cudaMemcpyAsync(h_buffer, d_buffer, test_size,
                          cudaMemcpyDeviceToHost, stream2);
        }
        cudaStreamSynchronize(stream2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float d2h_time;
        cudaEventElapsedTime(&d2h_time, start, stop);
        float d2h_bandwidth = (test_size * 10) / (d2h_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        printf("  H2D Bandwidth: %.2f GB/s\n", h2d_bandwidth);
        printf("  D2H Bandwidth: %.2f GB/s\n", d2h_bandwidth);

        // Test zero-copy access if mapped
        if (config.flags & cudaHostAllocMapped) {
            float* d_mapped_ptr;
            cudaHostGetDevicePointer(&d_mapped_ptr, h_buffer, 0);

            printf("  Zero-copy access enabled (device ptr: %p)\n", d_mapped_ptr);

            // Launch kernel that directly accesses host memory
            zero_copy_kernel<<<(test_size/sizeof(float) + 255)/256, 256>>>(
                d_mapped_ptr, test_size/sizeof(float));
            cudaDeviceSynchronize();

            printf("  Zero-copy kernel execution successful\n");
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Cleanup
    for (void* buffer : test_buffers) {
        manager.deallocate(buffer);
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_buffer);

    manager.print_statistics();
}

__global__ void zero_copy_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Direct access to host memory (zero-copy)
        data[tid] = sqrt(data[tid]) + 1.0f;
    }
}
```

###  **Bandwidth Optimization Strategies**

#### **Memory Transfer Pattern Analysis:**
```cpp
#include <vector>
#include <cstdio>

// Comprehensive bandwidth optimization techniques
class BandwidthOptimizer {
private:
    cudaDeviceProp device_props;
    float theoretical_bandwidth;

public:
    BandwidthOptimizer() {
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&device_props, device);

        // Calculate theoretical bandwidth
        theoretical_bandwidth = 2.0f * device_props.memoryClockRate *
                               (device_props.memoryBusWidth / 8) / 1.0e6;

        printf("=== Bandwidth Optimizer ===\n");
        printf("Device: %s\n", device_props.name);
        printf("Memory Clock Rate: %d kHz\n", device_props.memoryClockRate);
        printf("Memory Bus Width: %d bits\n", device_props.memoryBusWidth);
        printf("Theoretical Bandwidth: %.2f GB/s\n", theoretical_bandwidth);
        printf("===========================\n\n");
    }

    // Test different transfer sizes to find optimal chunk size
    void optimize_transfer_size() {
        printf("Optimizing Transfer Size:\n");

        const size_t max_size = 256 * 1024 * 1024; // 256MB
        const int num_iterations = 20;

        std::vector<size_t> test_sizes = {
            4 * 1024,           // 4KB
            64 * 1024,          // 64KB
            1024 * 1024,        // 1MB
            16 * 1024 * 1024,   // 16MB
            64 * 1024 * 1024,   // 64MB
            256 * 1024 * 1024   // 256MB
        };

        float *h_data, *d_data;
        cudaHostAlloc(&h_data, max_size, cudaHostAllocDefault);
        cudaMalloc(&d_data, max_size);

        // Initialize data
        for (size_t i = 0; i < max_size / sizeof(float); i++) {
            h_data[i] = i * 0.001f;
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        size_t optimal_size = 0;
        float best_bandwidth = 0.0f;

        for (size_t test_size : test_sizes) {
            // Measure H2D bandwidth
            cudaEventRecord(start);
            for (int i = 0; i < num_iterations; i++) {
                cudaMemcpyAsync(d_data, h_data, test_size,
                              cudaMemcpyHostToDevice, stream);
            }
            cudaStreamSynchronize(stream);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start, stop);
            float bandwidth = (test_size * num_iterations) /
                            (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

            printf("  Size: %8zu KB, Bandwidth: %6.2f GB/s (%.1f%% of theoretical)\n",
                   test_size / 1024, bandwidth,
                   (bandwidth / theoretical_bandwidth) * 100.0f);

            if (bandwidth > best_bandwidth) {
                best_bandwidth = bandwidth;
                optimal_size = test_size;
            }
        }

        printf("  Optimal transfer size: %zu KB (%.2f GB/s)\n\n",
               optimal_size / 1024, best_bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        cudaFreeHost(h_data);
        cudaFree(d_data);
    }

    // Test concurrent transfers with different stream counts
    void optimize_stream_count() {
        printf("Optimizing Stream Count for Concurrent Transfers:\n");

        const size_t transfer_size = 64 * 1024 * 1024; // 64MB per transfer
        const int max_streams = 8;

        for (int num_streams = 1; num_streams <= max_streams; num_streams++) {
            float total_bandwidth = test_concurrent_transfers(num_streams, transfer_size);
            printf("  %d streams: %.2f GB/s total bandwidth\n",
                   num_streams, total_bandwidth);
        }
        printf("\n");
    }

private:
    float test_concurrent_transfers(int num_streams, size_t transfer_size) {
        std::vector<float*> h_buffers(num_streams);
        std::vector<float*> d_buffers(num_streams);
        std::vector<cudaStream_t> streams(num_streams);

        // Allocate resources
        for (int i = 0; i < num_streams; i++) {
            cudaHostAlloc(&h_buffers[i], transfer_size, cudaHostAllocDefault);
            cudaMalloc(&d_buffers[i], transfer_size);
            cudaStreamCreate(&streams[i]);

            // Initialize data
            for (size_t j = 0; j < transfer_size / sizeof(float); j++) {
                h_buffers[i][j] = (i * 1000 + j) * 0.001f;
            }
        }

        // Measure concurrent transfers
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Launch all transfers concurrently
        for (int i = 0; i < num_streams; i++) {
            cudaMemcpyAsync(d_buffers[i], h_buffers[i], transfer_size,
                          cudaMemcpyHostToDevice, streams[i]);
        }

        // Wait for all transfers to complete
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        float total_bandwidth = (transfer_size * num_streams) /
                              (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        // Cleanup
        for (int i = 0; i < num_streams; i++) {
            cudaFreeHost(h_buffers[i]);
            cudaFree(d_buffers[i]);
            cudaStreamDestroy(streams[i]);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return total_bandwidth;
    }
};
```

###  **Advanced Transfer Patterns**

#### **Bidirectional Transfer Optimization:**
```cpp
#include <vector>
#include <cstdio>
#include <cmath>

// Forward declaration
__global__ void bidirectional_compute_kernel(float* data, int N, int iteration);

// Sophisticated bidirectional transfer patterns
class BidirectionalTransferManager {
private:
    struct TransferBuffer {
        float* h_buffer;
        float* d_buffer;
        size_t size;
        cudaStream_t upload_stream;
        cudaStream_t download_stream;
        cudaEvent_t upload_complete;
        cudaEvent_t download_complete;
    };

    std::vector<TransferBuffer> buffers;
    int num_buffers;

public:
    BidirectionalTransferManager(int buffer_count, size_t buffer_size)
        : num_buffers(buffer_count) {

        buffers.resize(buffer_count);

        for (int i = 0; i < buffer_count; i++) {
            // Allocate pinned host memory
            cudaHostAlloc(&buffers[i].h_buffer, buffer_size, cudaHostAllocDefault);

            // Allocate device memory
            cudaMalloc(&buffers[i].d_buffer, buffer_size);

            // Create dedicated streams for each direction
            cudaStreamCreate(&buffers[i].upload_stream);
            cudaStreamCreate(&buffers[i].download_stream);

            // Create events for synchronization
            cudaEventCreate(&buffers[i].upload_complete);
            cudaEventCreate(&buffers[i].download_complete);

            buffers[i].size = buffer_size;

            // Initialize with test data
            for (size_t j = 0; j < buffer_size / sizeof(float); j++) {
                buffers[i].h_buffer[j] = (i * 10000 + j) * 0.0001f;
            }
        }

        printf("BidirectionalTransferManager initialized with %d buffers\n", buffer_count);
    }

    // Demonstrate overlapped bidirectional transfers
    void demonstrate_bidirectional_overlap() {
        printf("=== Bidirectional Transfer Overlap ===\n");

        cudaEvent_t overall_start, overall_stop;
        cudaEventCreate(&overall_start);
        cudaEventCreate(&overall_stop);

        // Method 1: Sequential transfers
        printf("1. Sequential bidirectional transfers:\n");
        cudaEventRecord(overall_start);

        for (int i = 0; i < num_buffers; i++) {
            // Upload then download sequentially
            cudaMemcpy(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                      cudaMemcpyHostToDevice);
            cudaMemcpy(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                      cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(overall_stop);
        cudaEventSynchronize(overall_stop);

        float sequential_time;
        cudaEventElapsedTime(&sequential_time, overall_start, overall_stop);
        printf("   Sequential time: %.2f ms\n", sequential_time);

        // Method 2: Overlapped transfers using streams
        printf("2. Overlapped bidirectional transfers:\n");
        cudaEventRecord(overall_start);

        // Launch all uploads first
        for (int i = 0; i < num_buffers; i++) {
            cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                          cudaMemcpyHostToDevice, buffers[i].upload_stream);
            cudaEventRecord(buffers[i].upload_complete, buffers[i].upload_stream);
        }

        // Launch downloads that depend on uploads
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamWaitEvent(buffers[i].download_stream, buffers[i].upload_complete, 0);
            cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                          cudaMemcpyDeviceToHost, buffers[i].download_stream);
            cudaEventRecord(buffers[i].download_complete, buffers[i].download_stream);
        }

        // Wait for all downloads to complete
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamSynchronize(buffers[i].download_stream);
        }

        cudaEventRecord(overall_stop);
        cudaEventSynchronize(overall_stop);

        float overlapped_time;
        cudaEventElapsedTime(&overlapped_time, overall_start, overall_stop);
        printf("   Overlapped time: %.2f ms\n", overlapped_time);
        printf("   Speedup: %.2fx\n", sequential_time / overlapped_time);

        // Method 3: Advanced pipeline with computation
        printf("3. Pipeline with computation overlap:\n");
        pipeline_with_computation();

        cudaEventDestroy(overall_start);
        cudaEventDestroy(overall_stop);
    }

    void pipeline_with_computation() {
        cudaEvent_t pipeline_start, pipeline_stop;
        cudaEventCreate(&pipeline_start);
        cudaEventCreate(&pipeline_stop);

        cudaEventRecord(pipeline_start);

        // Complex pipeline: Upload -> Compute -> Download
        for (int iteration = 0; iteration < 3; iteration++) {
            for (int i = 0; i < num_buffers; i++) {
                // Stage 1: Upload data
                cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                              cudaMemcpyHostToDevice, buffers[i].upload_stream);
                cudaEventRecord(buffers[i].upload_complete, buffers[i].upload_stream);

                // Stage 2: Compute (depends on upload)
                cudaStreamWaitEvent(buffers[i].download_stream, buffers[i].upload_complete, 0);

                int num_elements = buffers[i].size / sizeof(float);
                bidirectional_compute_kernel<<<(num_elements + 255)/256, 256, 0,
                                             buffers[i].download_stream>>>(
                    buffers[i].d_buffer, num_elements, iteration);

                // Stage 3: Download result
                cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                              cudaMemcpyDeviceToHost, buffers[i].download_stream);
                cudaEventRecord(buffers[i].download_complete, buffers[i].download_stream);
            }

            // Wait for this iteration to complete before starting next
            for (int i = 0; i < num_buffers; i++) {
                cudaStreamSynchronize(buffers[i].download_stream);
            }

            printf("   Pipeline iteration %d complete\n", iteration + 1);
        }

        cudaEventRecord(pipeline_stop);
        cudaEventSynchronize(pipeline_stop);

        float pipeline_time;
        cudaEventElapsedTime(&pipeline_time, pipeline_start, pipeline_stop);
        printf("   Total pipeline time: %.2f ms\n", pipeline_time);

        cudaEventDestroy(pipeline_start);
        cudaEventDestroy(pipeline_stop);
    }

    // Analyze transfer bandwidth utilization
    void analyze_bandwidth_utilization() {
        printf("=== Bandwidth Utilization Analysis ===\n");

        const int num_measurements = 10;
        float total_bandwidth = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int measurement = 0; measurement < num_measurements; measurement++) {
            cudaEventRecord(start);

            // Concurrent bidirectional transfers
            for (int i = 0; i < num_buffers; i++) {
                cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                              cudaMemcpyHostToDevice, buffers[i].upload_stream);
                cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                              cudaMemcpyDeviceToHost, buffers[i].download_stream);
            }

            // Synchronize all transfers
            for (int i = 0; i < num_buffers; i++) {
                cudaStreamSynchronize(buffers[i].upload_stream);
                cudaStreamSynchronize(buffers[i].download_stream);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start, stop);

            // Calculate total bandwidth (upload + download)
            size_t total_bytes = buffers[0].size * num_buffers * 2; // *2 for bidirectional
            float bandwidth = total_bytes / (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            total_bandwidth += bandwidth;

            printf("   Measurement %d: %.2f GB/s\n", measurement + 1, bandwidth);
        }

        float avg_bandwidth = total_bandwidth / num_measurements;
        printf("   Average bandwidth: %.2f GB/s\n", avg_bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    ~BidirectionalTransferManager() {
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamSynchronize(buffers[i].upload_stream);
            cudaStreamSynchronize(buffers[i].download_stream);

            cudaFreeHost(buffers[i].h_buffer);
            cudaFree(buffers[i].d_buffer);
            cudaStreamDestroy(buffers[i].upload_stream);
            cudaStreamDestroy(buffers[i].download_stream);
            cudaEventDestroy(buffers[i].upload_complete);
            cudaEventDestroy(buffers[i].download_complete);
        }

        printf("BidirectionalTransferManager cleanup complete\n");
    }
};

__global__ void bidirectional_compute_kernel(float* data, int N, int iteration) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Different computation per iteration
        for (int i = 0; i < (iteration + 1) * 50; i++) {
            value = sin(value) + cos(value * 0.1f);
        }

        data[tid] = value;
    }
}
```

---

##  **Event-Driven Programming**

CUDA events provide precise synchronization control and performance measurement capabilities, enabling sophisticated coordination between streams and accurate timing analysis of GPU operations.

###  **Event Fundamentals and Types**

CUDA events serve as lightweight synchronization primitives that can be recorded in streams and waited upon, providing fine-grained control over execution dependencies.

#### **Comprehensive Event Management:**
```cpp
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cmath>

// Forward declaration
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

### ⏱ **Precision Timing and Performance Measurement**

Events provide the most accurate method for measuring GPU execution times, with sub-millisecond precision and minimal overhead.

#### **Advanced Timing Infrastructure:**
```cpp
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdio>

// Forward declarations
__global__ void complex_math_kernel(float* input, float* output, int N);
__global__ void simple_math_kernel(float* input, float* output, int N);

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

###  **Advanced Synchronization Patterns**

Events enable sophisticated synchronization patterns beyond basic stream coordination, including complex dependency graphs and multi-stage pipeline coordination.

#### **Event-Based Coordination Patterns:**
```cpp
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <functional>
#include <thread>
#include <chrono>
#include <cstdio>
#include <algorithm>

// Forward declarations
__global__ void initialization_kernel(float* data, int N);
__global__ void combine_kernel(float* data1, float* data2, float* output, int N);
__global__ void complex_math_kernel(float* input, float* output, int N);
__global__ void simple_math_kernel(float* input, float* output, int N);

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

---

##  **CUDA Graphs Deep Dive**

CUDA Graphs represent a paradigm shift from dynamic kernel launches to static execution graphs, enabling dramatic performance improvements for repetitive workloads by reducing launch overhead and enabling advanced optimizations.

###  **Graph Fundamentals and Architecture**

CUDA Graphs capture sequences of GPU operations into a static directed acyclic graph (DAG), allowing the CUDA runtime to optimize execution and minimize overhead.

#### **Comprehensive Graph Management System:**
```cpp
#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cmath>

// Forward declarations
__global__ void graph_stage1_kernel(float* input, float* output, int N);
__global__ void graph_stage2_kernel(float* input, float* output, int N);

// Advanced CUDA Graph management for production applications
class GraphManager {
private:
    struct GraphInfo {
        std::string name;
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        bool is_instantiated;
        bool is_captured;
        std::chrono::high_resolution_clock::time_point creation_time;
        int execution_count;
        float total_execution_time;
        std::vector<std::string> dependencies;
    };

    std::map<std::string, GraphInfo> graphs;
    std::map<std::string, cudaStream_t> capture_streams;

public:
    GraphManager() {
        printf("GraphManager initialized\n");
    }

    // Create a stream for graph capture
    cudaStream_t create_capture_stream(const std::string& stream_name) {
        if (capture_streams.find(stream_name) != capture_streams.end()) {
            printf("Stream '%s' already exists\n", stream_name.c_str());
            return capture_streams[stream_name];
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        capture_streams[stream_name] = stream;

        printf("Created capture stream '%s'\n", stream_name.c_str());
        return stream;
    }

    // Begin graph capture on a stream
    bool begin_capture(const std::string& graph_name, const std::string& stream_name,
                      cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) {

        if (graphs.find(graph_name) != graphs.end()) {
            printf("Graph '%s' already exists\n", graph_name.c_str());
            return false;
        }

        auto stream_it = capture_streams.find(stream_name);
        if (stream_it == capture_streams.end()) {
            printf("Stream '%s' not found\n", stream_name.c_str());
            return false;
        }

        // Initialize graph info
        GraphInfo info;
        info.name = graph_name;
        info.graph = nullptr;
        info.graph_exec = nullptr;
        info.is_instantiated = false;
        info.is_captured = false;
        info.creation_time = std::chrono::high_resolution_clock::now();
        info.execution_count = 0;
        info.total_execution_time = 0.0f;

        graphs[graph_name] = info;

        // Begin capture
        cudaError_t result = cudaStreamBeginCapture(stream_it->second, mode);
        if (result != cudaSuccess) {
            printf("Failed to begin capture for graph '%s': %s\n",
                   graph_name.c_str(), cudaGetErrorString(result));
            graphs.erase(graph_name);
            return false;
        }

        printf("Started capturing graph '%s' on stream '%s' (mode: %d)\n",
               graph_name.c_str(), stream_name.c_str(), mode);
        return true;
    }

    // End graph capture
    bool end_capture(const std::string& graph_name, const std::string& stream_name) {
        auto graph_it = graphs.find(graph_name);
        if (graph_it == graphs.end()) {
            printf("Graph '%s' not found\n", graph_name.c_str());
            return false;
        }

        auto stream_it = capture_streams.find(stream_name);
        if (stream_it == capture_streams.end()) {
            printf("Stream '%s' not found\n", stream_name.c_str());
            return false;
        }

        // End capture
        cudaError_t result = cudaStreamEndCapture(stream_it->second, &graph_it->second.graph);
        if (result != cudaSuccess) {
            printf("Failed to end capture for graph '%s': %s\n",
                   graph_name.c_str(), cudaGetErrorString(result));
            return false;
        }

        graph_it->second.is_captured = true;

        printf("Completed capturing graph '%s'\n", graph_name.c_str());
        return true;
    }

    // Instantiate a captured graph for execution
    bool instantiate_graph(const std::string& graph_name) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end() || !it->second.is_captured) {
            printf("Graph '%s' not found or not captured\n", graph_name.c_str());
            return false;
        }

        if (it->second.is_instantiated) {
            printf("Graph '%s' already instantiated\n", graph_name.c_str());
            return true;
        }

        // Instantiate the graph
        cudaError_t result = cudaGraphInstantiate(&it->second.graph_exec, it->second.graph,
                                                 nullptr, nullptr, 0);

        if (result != cudaSuccess) {
            printf("Failed to instantiate graph '%s': %s\n",
                   graph_name.c_str(), cudaGetErrorString(result));
            return false;
        }

        it->second.is_instantiated = true;

        printf("Instantiated graph '%s' for execution\n", graph_name.c_str());
        return true;
    }

    // Launch an instantiated graph
    bool launch_graph(const std::string& graph_name, cudaStream_t stream = 0) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end() || !it->second.is_instantiated) {
            printf("Graph '%s' not found or not instantiated\n", graph_name.c_str());
            return false;
        }

        // Time the execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);

        cudaError_t result = cudaGraphLaunch(it->second.graph_exec, stream);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        if (result != cudaSuccess) {
            printf("Failed to launch graph '%s': %s\n",
                   graph_name.c_str(), cudaGetErrorString(result));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }

        // Update execution statistics
        float execution_time;
        cudaEventElapsedTime(&execution_time, start, stop);

        it->second.execution_count++;
        it->second.total_execution_time += execution_time;

        printf("Executed graph '%s' in %.3f ms (execution #%d)\n",
               graph_name.c_str(), execution_time, it->second.execution_count);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return true;
    }

    // Update graph parameters (for dynamic graphs)
    bool update_graph_parameters(const std::string& graph_name,
                                const std::vector<cudaGraphNode_t>& nodes,
                                const std::vector<cudaKernelNodeParams>& new_params) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end() || !it->second.is_instantiated) {
            printf("Graph '%s' not found or not instantiated\n", graph_name.c_str());
            return false;
        }

        if (nodes.size() != new_params.size()) {
            printf("Mismatch between nodes and parameters count\n");
            return false;
        }

        for (size_t i = 0; i < nodes.size(); i++) {
            cudaError_t result = cudaGraphExecKernelNodeSetParams(
                it->second.graph_exec, nodes[i], &new_params[i]);

            if (result != cudaSuccess) {
                printf("Failed to update node %zu in graph '%s': %s\n",
                       i, graph_name.c_str(), cudaGetErrorString(result));
                return false;
            }
        }

        printf("Updated %zu nodes in graph '%s'\n", nodes.size(), graph_name.c_str());
        return true;
    }

    // Clone an existing graph
    bool clone_graph(const std::string& source_name, const std::string& target_name) {
        auto source_it = graphs.find(source_name);
        if (source_it == graphs.end() || !source_it->second.is_captured) {
            printf("Source graph '%s' not found or not captured\n", source_name.c_str());
            return false;
        }

        if (graphs.find(target_name) != graphs.end()) {
            printf("Target graph '%s' already exists\n", target_name.c_str());
            return false;
        }

        // Clone the graph
        GraphInfo clone_info = source_it->second;
        clone_info.name = target_name;
        clone_info.graph_exec = nullptr;
        clone_info.is_instantiated = false;
        clone_info.execution_count = 0;
        clone_info.total_execution_time = 0.0f;
        clone_info.creation_time = std::chrono::high_resolution_clock::now();

        cudaError_t result = cudaGraphClone(&clone_info.graph, source_it->second.graph);
        if (result != cudaSuccess) {
            printf("Failed to clone graph '%s': %s\n",
                   source_name.c_str(), cudaGetErrorString(result));
            return false;
        }

        graphs[target_name] = clone_info;

        printf("Cloned graph '%s' to '%s'\n", source_name.c_str(), target_name.c_str());
        return true;
    }

    // Get graph execution statistics
    void print_graph_statistics(const std::string& graph_name) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end()) {
            printf("Graph '%s' not found\n", graph_name.c_str());
            return;
        }

        const GraphInfo& info = it->second;

        printf("=== Graph '%s' Statistics ===\n", graph_name.c_str());
        printf("Status: %s%s\n",
               info.is_captured ? "Captured " : "Not captured ",
               info.is_instantiated ? "Instantiated" : "Not instantiated");
        printf("Executions: %d\n", info.execution_count);

        if (info.execution_count > 0) {
            float avg_time = info.total_execution_time / info.execution_count;
            printf("Total execution time: %.3f ms\n", info.total_execution_time);
            printf("Average execution time: %.3f ms\n", avg_time);

            // Calculate throughput if we have timing data
            auto current_time = std::chrono::high_resolution_clock::now();
            auto lifetime = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - info.creation_time);

            if (lifetime.count() > 0) {
                float executions_per_sec = (info.execution_count * 1000.0f) / lifetime.count();
                printf("Execution rate: %.2f executions/second\n", executions_per_sec);
            }
        }

        printf("Dependencies: %zu\n", info.dependencies.size());
        printf("===============================\n");
    }

    // Print all graphs statistics
    void print_all_statistics() {
        printf("=== GraphManager Statistics ===\n");
        printf("Total graphs: %zu\n", graphs.size());
        printf("Capture streams: %zu\n", capture_streams.size());

        int captured = 0, instantiated = 0, total_executions = 0;
        float total_time = 0.0f;

        for (const auto& pair : graphs) {
            if (pair.second.is_captured) captured++;
            if (pair.second.is_instantiated) instantiated++;
            total_executions += pair.second.execution_count;
            total_time += pair.second.total_execution_time;
        }

        printf("Captured graphs: %d\n", captured);
        printf("Instantiated graphs: %d\n", instantiated);
        printf("Total executions: %d\n", total_executions);
        printf("Total execution time: %.3f ms\n", total_time);

        if (total_executions > 0) {
            printf("Average execution time: %.3f ms\n", total_time / total_executions);
        }

        printf("===============================\n");
    }

    // Destroy a graph
    void destroy_graph(const std::string& graph_name) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end()) {
            printf("Graph '%s' not found\n", graph_name.c_str());
            return;
        }

        // Clean up graph resources
        if (it->second.graph_exec != nullptr) {
            cudaGraphExecDestroy(it->second.graph_exec);
        }

        if (it->second.graph != nullptr) {
            cudaGraphDestroy(it->second.graph);
        }

        graphs.erase(it);
        printf("Destroyed graph '%s'\n", graph_name.c_str());
    }

    ~GraphManager() {
        printf("Destroying GraphManager...\n");

        // Clean up all graphs
        for (auto& pair : graphs) {
            if (pair.second.graph_exec != nullptr) {
                cudaGraphExecDestroy(pair.second.graph_exec);
            }
            if (pair.second.graph != nullptr) {
                cudaGraphDestroy(pair.second.graph);
            }
            printf("Destroyed graph '%s'\n", pair.first.c_str());
        }

        // Clean up capture streams
        for (auto& pair : capture_streams) {
            cudaStreamDestroy(pair.second);
            printf("Destroyed stream '%s'\n", pair.first.c_str());
        }

        printf("GraphManager cleanup complete\n");
    }
};

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

###  **Advanced Graph Patterns and Optimization**

#### **Dynamic Graph Updates and Parameter Modification:**
```cpp
#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <cmath>
#include <limits>
#include <algorithm>

// Forward declarations
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

###  **Production Graph Optimization Strategies**

#### **Enterprise-Grade Graph Management:**
```cpp
#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <limits>
#include <algorithm>
#include <cmath>

// Forward declaration
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
```

---

##  **Advanced Stream Patterns**

Beyond basic stream operations, CUDA enables sophisticated coordination patterns that maximize GPU utilization through complex producer-consumer relationships, pipeline architectures, and dynamic load balancing strategies.

###  **Producer-Consumer Patterns**

Producer-consumer patterns enable efficient data flow management where different components generate and consume data at potentially different rates, requiring sophisticated buffering and synchronization strategies.

#### **Multi-Buffer Producer-Consumer System:**
```cpp
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cstdio>

// Forward declarations
__global__ void producer_kernel(float* output, int N, int item_id);
__global__ void consumer_kernel(float* input, int N, int sequence);

// Advanced producer-consumer pattern with dynamic buffering
template<typename T>
class StreamProducerConsumer {
private:
    struct BufferSlot {
        T* device_buffer;
        cudaEvent_t ready_event;
        cudaEvent_t consumed_event;
        bool is_producer_ready;
        bool is_consumer_ready;
        size_t data_size;
        int sequence_number;
    };

    std::vector<BufferSlot> buffer_ring;
    std::queue<int> producer_queue;
    std::queue<int> consumer_queue;

    cudaStream_t producer_stream;
    cudaStream_t consumer_stream;

    size_t buffer_size;
    int num_buffers;
    int producer_index;
    int consumer_index;
    int sequence_counter;

    std::mutex queue_mutex;
    std::condition_variable producer_cv;
    std::condition_variable consumer_cv;

    bool shutdown_requested;

public:
    StreamProducerConsumer(size_t buf_size, int num_bufs = 4)
        : buffer_size(buf_size), num_buffers(num_bufs),
          producer_index(0), consumer_index(0), sequence_counter(0),
          shutdown_requested(false) {

        printf("Initializing StreamProducerConsumer (buffers: %d, size: %zu bytes)\n",
               num_buffers, buffer_size);

        // Create streams
        cudaStreamCreate(&producer_stream);
        cudaStreamCreate(&consumer_stream);

        // Initialize buffer ring
        buffer_ring.resize(num_buffers);

        for (int i = 0; i < num_buffers; i++) {
            BufferSlot& slot = buffer_ring[i];

            // Allocate device memory
            cudaMalloc(&slot.device_buffer, buffer_size);

            // Create events
            cudaEventCreate(&slot.ready_event);
            cudaEventCreate(&slot.consumed_event);

            // Initialize state
            slot.is_producer_ready = false;
            slot.is_consumer_ready = false;
            slot.data_size = 0;
            slot.sequence_number = -1;

            // Initially available for producer
            producer_queue.push(i);
        }

        printf("StreamProducerConsumer initialized successfully\n");
    }

    // Producer interface - get next available buffer
    T* get_producer_buffer(size_t data_size, int& buffer_id) {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Wait for available buffer
        producer_cv.wait(lock, [this] {
            return !producer_queue.empty() || shutdown_requested;
        });

        if (shutdown_requested) {
            buffer_id = -1;
            return nullptr;
        }

        buffer_id = producer_queue.front();
        producer_queue.pop();

        BufferSlot& slot = buffer_ring[buffer_id];

        // Wait for the consumer to finish using this buffer
        // Note: First use sequence_number is -1, so no event wait needed
        if (slot.sequence_number != -1) {
            cudaStreamWaitEvent(producer_stream, slot.consumed_event, 0);
        }

        slot.data_size = data_size;
        slot.sequence_number = sequence_counter++;
        slot.is_producer_ready = false;

        printf("Producer acquired buffer %d (sequence: %d)\n",
               buffer_id, slot.sequence_number);

        return slot.device_buffer;
    }

    // Producer interface - mark buffer as ready for consumption
    void submit_producer_buffer(int buffer_id) {
        if (buffer_id < 0 || buffer_id >= num_buffers) {
            printf("Invalid buffer ID: %d\n", buffer_id);
            return;
        }

        BufferSlot& slot = buffer_ring[buffer_id];

        // Record ready event
        cudaEventRecord(slot.ready_event, producer_stream);

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            slot.is_producer_ready = true;
            consumer_queue.push(buffer_id);
        }

        consumer_cv.notify_one();

        printf("Producer submitted buffer %d (sequence: %d)\n",
               buffer_id, slot.sequence_number);
    }

    // Consumer interface - get next ready buffer
    T* get_consumer_buffer(int& buffer_id, size_t& data_size, int& sequence) {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Wait for ready buffer
        consumer_cv.wait(lock, [this] {
            return !consumer_queue.empty() || shutdown_requested;
        });

        if (shutdown_requested) {
            buffer_id = -1;
            return nullptr;
        }

        buffer_id = consumer_queue.front();
        consumer_queue.pop();

        BufferSlot& slot = buffer_ring[buffer_id];

        // Wait for producer to finish
        cudaStreamWaitEvent(consumer_stream, slot.ready_event, 0);

        data_size = slot.data_size;
        sequence = slot.sequence_number;

        printf("Consumer acquired buffer %d (sequence: %d, size: %zu)\n",
               buffer_id, sequence, data_size);

        return slot.device_buffer;
    }

    // Consumer interface - mark buffer as consumed
    void release_consumer_buffer(int buffer_id) {
        if (buffer_id < 0 || buffer_id >= num_buffers) {
            printf("Invalid buffer ID: %d\n", buffer_id);
            return;
        }

        BufferSlot& slot = buffer_ring[buffer_id];

        // Record consumed event
        cudaEventRecord(slot.consumed_event, consumer_stream);

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            slot.is_consumer_ready = false;
            slot.is_producer_ready = false;
            producer_queue.push(buffer_id);
        }

        producer_cv.notify_one();

        printf("Consumer released buffer %d (sequence: %d)\n",
               buffer_id, slot.sequence_number);
    }

    // Get streams for external operations
    cudaStream_t get_producer_stream() const { return producer_stream; }
    cudaStream_t get_consumer_stream() const { return consumer_stream; }

    // Shutdown the system gracefully
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            shutdown_requested = true;
        }

        producer_cv.notify_all();
        consumer_cv.notify_all();

        printf("StreamProducerConsumer shutdown initiated\n");
    }

    // Get system statistics
    void print_statistics() {
        printf("=== StreamProducerConsumer Statistics ===\n");
        printf("Buffer configuration: %d buffers, %zu bytes each\n",
               num_buffers, buffer_size);
        printf("Total sequences processed: %d\n", sequence_counter);

        int available_for_producer = 0;
        int available_for_consumer = 0;

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            available_for_producer = producer_queue.size();
            available_for_consumer = consumer_queue.size();
        }

        printf("Currently available for producer: %d\n", available_for_producer);
        printf("Currently available for consumer: %d\n", available_for_consumer);
        printf("Shutdown requested: %s\n", shutdown_requested ? "yes" : "no");
        printf("========================================\n");
    }

    ~StreamProducerConsumer() {
        printf("Destroying StreamProducerConsumer...\n");

        shutdown();

        // Cleanup buffers and events
        for (auto& slot : buffer_ring) {
            if (slot.device_buffer) {
                cudaFree(slot.device_buffer);
            }
            cudaEventDestroy(slot.ready_event);
            cudaEventDestroy(slot.consumed_event);
        }

        // Destroy streams
        cudaStreamDestroy(producer_stream);
        cudaStreamDestroy(consumer_stream);

        printf("StreamProducerConsumer cleanup complete\n");
    }
};

// Producer function for demonstration
template<typename T>
void producer_worker(StreamProducerConsumer<T>& system, int num_items) {
    printf("Producer worker started (will produce %d items)\n", num_items);

    for (int i = 0; i < num_items; i++) {
        int buffer_id;
        size_t data_size = sizeof(T) * 1024; // 1K elements

        T* buffer = system.get_producer_buffer(data_size, buffer_id);
        if (!buffer) {
            printf("Producer: Failed to get buffer, shutting down\n");
            break;
        }

        // Simulate data generation work
        producer_kernel<<<64, 16, 0, system.get_producer_stream()>>>(
            buffer, 1024, i);

        // Submit for consumption
        system.submit_producer_buffer(buffer_id);

        // Simulate variable production rate
        std::this_thread::sleep_for(std::chrono::milliseconds(10 + (i % 20)));
    }

    printf("Producer worker completed\n");
}

// Consumer function for demonstration
template<typename T>
void consumer_worker(StreamProducerConsumer<T>& system, int num_items) {
    printf("Consumer worker started (will consume %d items)\n", num_items);

    for (int i = 0; i < num_items; i++) {
        int buffer_id, sequence;
        size_t data_size;

        T* buffer = system.get_consumer_buffer(buffer_id, data_size, sequence);
        if (!buffer) {
            printf("Consumer: Failed to get buffer, shutting down\n");
            break;
        }

        // Simulate data processing work
        consumer_kernel<<<64, 16, 0, system.get_consumer_stream()>>>(
            buffer, data_size / sizeof(T), sequence);

        // Release buffer
        system.release_consumer_buffer(buffer_id);

        // Simulate variable consumption rate
        std::this_thread::sleep_for(std::chrono::milliseconds(15 + (i % 15)));
    }

    printf("Consumer worker completed\n");
}

// Demonstrate producer-consumer pattern
void demonstrate_producer_consumer_pattern() {
    printf("=== Producer-Consumer Pattern Demo ===\n");

    const int num_items = 20;
    StreamProducerConsumer<float> system(1024 * sizeof(float), 6);

    // Launch producer and consumer in separate threads
    std::thread producer_thread(producer_worker<float>, std::ref(system), num_items);
    std::thread consumer_thread(consumer_worker<float>, std::ref(system), num_items);

    // Monitor system for a while
    for (int i = 0; i < 10; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        system.print_statistics();
    }

    // Wait for completion
    producer_thread.join();
    consumer_thread.join();

    // Final statistics
    system.print_statistics();
}

__global__ void producer_kernel(float* output, int N, int item_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = item_id + tid * 0.001f;
    }

    if (tid == 0) {
        printf("GPU Producer: Generated item %d\n", item_id);
    }
}

__global__ void consumer_kernel(float* input, int N, int sequence) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float value = input[tid];
        // Simulate processing
        input[tid] = value * 2.0f + 1.0f;
    }

    if (tid == 0) {
        printf("GPU Consumer: Processed sequence %d\n", sequence);
    }
}
```

###  **Pipeline Architecture Patterns**

Stream-based pipelines enable complex multi-stage processing where each stage can operate independently and concurrently, maximizing GPU utilization and throughput.

#### **Multi-Stage Processing Pipeline:**
```cpp
#include <vector>
#include <string>
#include <functional>
#include <cstdio>

// Forward declarations
__global__ void pipeline_preprocess_kernel(float* input, float* output, int N);
__global__ void pipeline_compute_kernel(float* input, float* output, int N);
__global__ void pipeline_postprocess_kernel(float* input, float* output, int N);
__global__ void pipeline_generic_kernel(float* input, float* output, int N, int stage_id);

// Sophisticated multi-stage pipeline with dynamic load balancing
class StreamPipeline {
private:
    struct PipelineStage {
        std::string name;
        cudaStream_t stream;
        std::function<void(float*, float*, int, cudaStream_t)> process_func;
        float* input_buffer;
        float* output_buffer;
        cudaEvent_t stage_complete;
        int buffer_size;
        float avg_processing_time;
        int completed_batches;
    };

    std::vector<PipelineStage> stages;
    std::vector<float*> intermediate_buffers;
    int num_stages;
    int buffer_elements;
    bool is_initialized;

public:
    StreamPipeline(int num_pipeline_stages, int elements_per_buffer)
        : num_stages(num_pipeline_stages), buffer_elements(elements_per_buffer),
          is_initialized(false) {

        printf("Initializing StreamPipeline with %d stages (%d elements per buffer)\n",
               num_stages, buffer_elements);

        stages.resize(num_stages);
        intermediate_buffers.resize(num_stages + 1);

        // Allocate intermediate buffers
        for (int i = 0; i <= num_stages; i++) {
            cudaMalloc(&intermediate_buffers[i], buffer_elements * sizeof(float));
            printf("Allocated buffer %d: %p\n", i, intermediate_buffers[i]);
        }

        // Initialize pipeline stages
        for (int i = 0; i < num_stages; i++) {
            PipelineStage& stage = stages[i];
            stage.name = "Stage_" + std::to_string(i);

            cudaStreamCreate(&stage.stream);
            cudaEventCreate(&stage.stage_complete);

            stage.input_buffer = intermediate_buffers[i];
            stage.output_buffer = intermediate_buffers[i + 1];
            stage.buffer_size = buffer_elements;
            stage.avg_processing_time = 0.0f;
            stage.completed_batches = 0;

            printf("Initialized %s: input=%p, output=%p\n",
                   stage.name.c_str(), stage.input_buffer, stage.output_buffer);
        }

        setup_default_processing_functions();
        is_initialized = true;

        printf("StreamPipeline initialization complete\n");
    }

    // Set custom processing function for a stage
    void set_stage_processor(int stage_id,
                           std::function<void(float*, float*, int, cudaStream_t)> processor,
                           const std::string& stage_name = "") {
        if (stage_id < 0 || stage_id >= num_stages) {
            printf("Invalid stage ID: %d\n", stage_id);
            return;
        }

        stages[stage_id].process_func = processor;

        if (!stage_name.empty()) {
            stages[stage_id].name = stage_name;
        }

        printf("Set custom processor for stage %d (%s)\n",
               stage_id, stages[stage_id].name.c_str());
    }

    // Execute pipeline on input data
    void execute_pipeline(float* input_data, float* output_data,
                         bool measure_performance = true) {
        if (!is_initialized) {
            printf("Pipeline not initialized\n");
            return;
        }

        printf("=== Executing Pipeline ===\n");

        // Copy input data to first buffer
        cudaMemcpy(intermediate_buffers[0], input_data,
                  buffer_elements * sizeof(float), cudaMemcpyHostToDevice);

        std::vector<cudaEvent_t> stage_timers;
        if (measure_performance) {
            stage_timers.resize(num_stages * 2); // start and stop for each stage
            for (auto& event : stage_timers) {
                cudaEventCreate(&event);
            }
        }

        // Execute each stage
        for (int i = 0; i < num_stages; i++) {
            PipelineStage& stage = stages[i];

            printf("Executing %s...\n", stage.name.c_str());

            // Wait for previous stage if not the first
            if (i > 0) {
                cudaStreamWaitEvent(stage.stream, stages[i-1].stage_complete, 0);
            }

            // Record timing start
            if (measure_performance) {
                cudaEventRecord(stage_timers[i*2], stage.stream);
            }

            // Execute stage processing
            stage.process_func(stage.input_buffer, stage.output_buffer,
                             stage.buffer_size, stage.stream);

            // Record completion event
            cudaEventRecord(stage.stage_complete, stage.stream);

            // Record timing end
            if (measure_performance) {
                cudaEventRecord(stage_timers[i*2 + 1], stage.stream);
            }
        }

        // Wait for final stage and copy result
        cudaStreamSynchronize(stages[num_stages-1].stream);
        cudaMemcpy(output_data, intermediate_buffers[num_stages],
                  buffer_elements * sizeof(float), cudaMemcpyDeviceToHost);

        // Process performance measurements
        if (measure_performance) {
            printf("\nStage Performance Analysis:\n");

            for (int i = 0; i < num_stages; i++) {
                float stage_time;
                cudaEventElapsedTime(&stage_time, stage_timers[i*2], stage_timers[i*2 + 1]);

                stages[i].avg_processing_time =
                    (stages[i].avg_processing_time * stages[i].completed_batches + stage_time) /
                    (stages[i].completed_batches + 1);

                stages[i].completed_batches++;

                printf("  %s: %.3f ms (avg: %.3f ms over %d batches)\n",
                       stages[i].name.c_str(), stage_time,
                       stages[i].avg_processing_time, stages[i].completed_batches);
            }

            // Cleanup timing events
            for (auto& event : stage_timers) {
                cudaEventDestroy(event);
            }
        }

        printf("Pipeline execution complete\n");
    }

    // Execute pipeline with multiple batches for throughput analysis
    void execute_batched_pipeline(float** input_batches, float** output_batches,
                                 int num_batches, bool overlap_execution = true) {
        printf("=== Batched Pipeline Execution ===\n");
        printf("Processing %d batches (overlap: %s)\n",
               num_batches, overlap_execution ? "enabled" : "disabled");

        if (!overlap_execution) {
            // Simple sequential execution
            for (int batch = 0; batch < num_batches; batch++) {
                printf("Processing batch %d/%d\n", batch + 1, num_batches);
                execute_pipeline(input_batches[batch], output_batches[batch], false);
            }
            return;
        }

        // Overlapped execution for maximum throughput
        cudaEvent_t batch_start, batch_end;
        cudaEventCreate(&batch_start);
        cudaEventCreate(&batch_end);

        cudaEventRecord(batch_start);

        for (int batch = 0; batch < num_batches; batch++) {
            printf("Starting batch %d/%d\n", batch + 1, num_batches);

            // Copy input data asynchronously
            cudaMemcpyAsync(intermediate_buffers[0], input_batches[batch],
                           buffer_elements * sizeof(float),
                           cudaMemcpyHostToDevice, stages[0].stream);

            // Execute pipeline stages with proper dependencies
            for (int i = 0; i < num_stages; i++) {
                PipelineStage& stage = stages[i];

                // Wait for previous stage
                if (i > 0) {
                    cudaStreamWaitEvent(stage.stream, stages[i-1].stage_complete, 0);
                }

                // Execute processing
                stage.process_func(stage.input_buffer, stage.output_buffer,
                                 stage.buffer_size, stage.stream);

                // Record completion
                cudaEventRecord(stage.stage_complete, stage.stream);
            }

            // Copy output data asynchronously
            cudaMemcpyAsync(output_batches[batch], intermediate_buffers[num_stages],
                           buffer_elements * sizeof(float),
                           cudaMemcpyDeviceToHost, stages[num_stages-1].stream);
        }

        // Wait for all batches to complete
        for (int i = 0; i < num_stages; i++) {
            cudaStreamSynchronize(stages[i].stream);
        }

        cudaEventRecord(batch_end);
        cudaEventSynchronize(batch_end);

        float total_time;
        cudaEventElapsedTime(&total_time, batch_start, batch_end);

        printf("Batched execution complete:\n");
        printf("  Total time: %.3f ms\n", total_time);
        printf("  Time per batch: %.3f ms\n", total_time / num_batches);
        printf("  Throughput: %.2f batches/second\n", (num_batches * 1000.0f) / total_time);

        cudaEventDestroy(batch_start);
        cudaEventDestroy(batch_end);
    }

    // Analyze pipeline bottlenecks
    void analyze_pipeline_bottlenecks() {
        printf("=== Pipeline Bottleneck Analysis ===\n");

        if (stages[0].completed_batches == 0) {
            printf("No execution data available. Run pipeline first.\n");
            return;
        }

        // Find slowest stage
        float max_time = 0.0f;
        int bottleneck_stage = -1;

        printf("Stage performance summary:\n");
        for (int i = 0; i < num_stages; i++) {
            printf("  %s: %.3f ms avg (%.2f%% of total)\n",
                   stages[i].name.c_str(), stages[i].avg_processing_time,
                   (stages[i].avg_processing_time / get_total_pipeline_time()) * 100.0f);

            if (stages[i].avg_processing_time > max_time) {
                max_time = stages[i].avg_processing_time;
                bottleneck_stage = i;
            }
        }

        if (bottleneck_stage >= 0) {
            printf("\nBottleneck identified: %s (%.3f ms)\n",
                   stages[bottleneck_stage].name.c_str(), max_time);

            // Provide optimization suggestions
            printf("Optimization suggestions:\n");
            printf("  - Consider parallelizing %s across multiple streams\n",
                   stages[bottleneck_stage].name.c_str());
            printf("  - Optimize kernel configuration for %s\n",
                   stages[bottleneck_stage].name.c_str());
            printf("  - Check if %s can be split into smaller sub-stages\n",
                   stages[bottleneck_stage].name.c_str());
        }

        // Calculate pipeline efficiency
        float theoretical_max_throughput = 1000.0f / max_time; // batches/second
        float actual_throughput = 1000.0f / get_total_pipeline_time();
        float efficiency = (actual_throughput / theoretical_max_throughput) * 100.0f;

        printf("\nPipeline efficiency: %.1f%%\n", efficiency);
        printf("Theoretical max throughput: %.2f batches/second\n", theoretical_max_throughput);
        printf("Actual throughput: %.2f batches/second\n", actual_throughput);

        printf("=====================================\n");
    }

    // Get pipeline statistics
    void print_pipeline_statistics() {
        printf("=== Pipeline Statistics ===\n");
        printf("Stages: %d\n", num_stages);
        printf("Buffer size: %d elements\n", buffer_elements);
        printf("Total pipeline time: %.3f ms\n", get_total_pipeline_time());

        printf("Individual stage statistics:\n");
        for (int i = 0; i < num_stages; i++) {
            const PipelineStage& stage = stages[i];
            printf("  %s:\n", stage.name.c_str());
            printf("    Completed batches: %d\n", stage.completed_batches);
            printf("    Average time: %.3f ms\n", stage.avg_processing_time);
            printf("    Stream: %p\n", stage.stream);
        }
        printf("==========================\n");
    }

private:
    void setup_default_processing_functions() {
        // Stage 0: Data preprocessing
        set_stage_processor(0, [](float* input, float* output, int N, cudaStream_t stream) {
            pipeline_preprocess_kernel<<<(N+255)/256, 256, 0, stream>>>(input, output, N);
        }, "Preprocessing");

        // Stage 1: Main computation
        if (num_stages > 1) {
            set_stage_processor(1, [](float* input, float* output, int N, cudaStream_t stream) {
                pipeline_compute_kernel<<<(N+255)/256, 256, 0, stream>>>(input, output, N);
            }, "MainCompute");
        }

        // Stage 2: Post-processing
        if (num_stages > 2) {
            set_stage_processor(2, [](float* input, float* output, int N, cudaStream_t stream) {
                pipeline_postprocess_kernel<<<(N+255)/256, 256, 0, stream>>>(input, output, N);
            }, "Postprocessing");
        }

        // Additional stages get generic processing
        for (int i = 3; i < num_stages; i++) {
            set_stage_processor(i, [i](float* input, float* output, int N, cudaStream_t stream) {
                pipeline_generic_kernel<<<(N+255)/256, 256, 0, stream>>>(input, output, N, i);
            }, "GenericStage_" + std::to_string(i));
        }
    }

    float get_total_pipeline_time() {
        float total = 0.0f;
        for (const auto& stage : stages) {
            total += stage.avg_processing_time;
        }
        return total;
    }

public:
    ~StreamPipeline() {
        printf("Destroying StreamPipeline...\n");

        // Cleanup streams and events
        for (auto& stage : stages) {
            cudaStreamDestroy(stage.stream);
            cudaEventDestroy(stage.stage_complete);
        }

        // Cleanup buffers
        for (auto buffer : intermediate_buffers) {
            cudaFree(buffer);
        }

        printf("StreamPipeline cleanup complete\n");
    }
};

// Demonstrate advanced pipeline patterns
void demonstrate_pipeline_patterns() {
    printf("=== Pipeline Patterns Demonstration ===\n");

    const int buffer_size = 1024 * 1024; // 1M elements
    const int num_batches = 5;

    // Create pipeline with 4 stages
    StreamPipeline pipeline(4, buffer_size);

    // Prepare test data
    std::vector<float*> input_batches(num_batches);
    std::vector<float*> output_batches(num_batches);

    for (int i = 0; i < num_batches; i++) {
        input_batches[i] = new float[buffer_size];
        output_batches[i] = new float[buffer_size];

        // Initialize input data
        for (int j = 0; j < buffer_size; j++) {
            input_batches[i][j] = i * 1000.0f + j * 0.001f;
        }
    }

    printf("\n1. Single Pipeline Execution:\n");
    pipeline.execute_pipeline(input_batches[0], output_batches[0], true);

    printf("\n2. Sequential Batch Processing:\n");
    pipeline.execute_batched_pipeline(input_batches.data(), output_batches.data(),
                                    num_batches, false);

    printf("\n3. Overlapped Batch Processing:\n");
    pipeline.execute_batched_pipeline(input_batches.data(), output_batches.data(),
                                    num_batches, true);

    printf("\n4. Pipeline Analysis:\n");
    pipeline.print_pipeline_statistics();
    pipeline.analyze_pipeline_bottlenecks();

    // Cleanup
    for (int i = 0; i < num_batches; i++) {
        delete[] input_batches[i];
        delete[] output_batches[i];
    }
}

// Pipeline kernel implementations
__global__ void pipeline_preprocess_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Normalization and basic preprocessing
        output[tid] = (input[tid] - 128.0f) / 255.0f;
    }
}

__global__ void pipeline_compute_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Main computation - complex mathematical operations
        float value = input[tid];
        for (int i = 0; i < 10; i++) {
            value = sin(value) + cos(value * 0.5f);
        }
        output[tid] = value;
    }
}

__global__ void pipeline_postprocess_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Post-processing - scaling and clamping
        float value = input[tid] * 255.0f + 128.0f;
        output[tid] = fmaxf(0.0f, fminf(255.0f, value));
    }
}

__global__ void pipeline_generic_kernel(float* input, float* output, int N, int stage_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Generic processing based on stage ID
        output[tid] = input[tid] * (stage_id + 1) + 0.1f;
    }
}
```

###  **Dynamic Load Balancing**

Advanced stream patterns can dynamically distribute work across multiple streams based on real-time performance characteristics and system load.

#### **Adaptive Stream Load Balancer:**
```cpp
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <thread>
#include <functional>
#include <string>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdio>

// Forward declarations
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

// Demonstrate adaptive load balancing
void demonstrate_adaptive_load_balancing() {
    printf("=== Adaptive Load Balancing Demo ===\n");

    const int num_workers = 4;
    const int num_tasks = 20;

    AdaptiveStreamBalancer balancer(num_workers);

    // Create various task types with different computational loads
    std::vector<std::function<void(cudaStream_t, int)>> tasks;
    std::vector<int> task_data;

    for (int i = 0; i < num_tasks; i++) {
        task_data.push_back(i);

        // Create tasks with varying computational complexity
        if (i % 3 == 0) {
            // Light task
            tasks.emplace_back([](cudaStream_t stream, int data) {
                light_compute_kernel<<<64, 64, 0, stream>>>(data);
            });
        } else if (i % 3 == 1) {
            // Medium task
            tasks.emplace_back([](cudaStream_t stream, int data) {
                medium_compute_kernel<<<128, 128, 0, stream>>>(data);
            });
        } else {
            // Heavy task
            tasks.emplace_back([](cudaStream_t stream, int data) {
                heavy_compute_kernel<<<256, 256, 0, stream>>>(data);
            });
        }
    }

    printf("\n1. Submitting Mixed Workload:\n");
    balancer.submit_task_batch(tasks, task_data);

    printf("\n2. Monitoring Load During Execution:\n");
    for (int i = 0; i < 5; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        balancer.rebalance_load();
    }

    printf("\n3. Waiting for Completion:\n");
    balancer.wait_for_completion();

    printf("\n4. Final Statistics:\n");
    balancer.print_comprehensive_statistics();
}

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

// Comprehensive demonstration of advanced stream patterns
void demonstrate_comprehensive_advanced_patterns() {
    printf("=== Comprehensive Advanced Stream Patterns Demo ===\n");

    printf("\n1. Producer-Consumer Pattern:\n");
    demonstrate_producer_consumer_pattern();

    printf("\n2. Pipeline Architecture:\n");
    demonstrate_pipeline_patterns();

    printf("\n3. Adaptive Load Balancing:\n");
    demonstrate_adaptive_load_balancing();

    printf("\nAdvanced stream patterns demonstration complete!\n");
}
```

---

## 8.  Nsight Debugging Tips

- Use **Nsight Systems** to visualize:
  - Stream timelines
  - Overlap of memcopy and kernels
- Identify serialization caused by:
  - Shared resources
  - Host sync calls (`cudaDeviceSynchronize()`)

