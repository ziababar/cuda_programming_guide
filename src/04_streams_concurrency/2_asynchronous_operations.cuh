#ifndef ASYNCHRONOUS_OPERATIONS_H
#define ASYNCHRONOUS_OPERATIONS_H

#include <cstdio>
#include <vector>
#include <queue>
#include <map>
#include <string>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

// Forward declarations of kernels used in this module
__global__ void complex_processing_kernel(float* input, float* output, int N);
__global__ void stage1_kernel(float* input, float* output, int N);
__global__ void stage2_kernel(float* input, float* output, int N);
__global__ void parallel_work_kernel(int stream_id);
__global__ void join_work_kernel();
__global__ void pipeline_stage_kernel(int stage, int iteration);
__global__ void initial_processing_kernel();
__global__ void parallel_processing_kernel(int branch_id);
__global__ void aggregation_kernel();


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


// Kernel implementations for synchronization demo
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

#endif // ASYNCHRONOUS_OPERATIONS_H
