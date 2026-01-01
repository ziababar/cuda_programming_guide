#ifndef STREAM_FUNDAMENTALS_H
#define STREAM_FUNDAMENTALS_H

#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

// Kernel declarations
__global__ void simple_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Simple computation to demonstrate stream behavior
        data[tid] = tid * 2.0f + sin(tid * 0.01f);
    }
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

#endif // STREAM_FUNDAMENTALS_H
