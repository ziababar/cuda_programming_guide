#  Asynchronous Operations

Asynchronous execution is the cornerstone of high-performance GPU programming, enabling overlapped computation, memory transfer concurrency, and sophisticated pipeline orchestration.

##  Compute-Transfer Overlap

The ability to overlap computation with memory transfers is one of the most powerful features of CUDA streams, often yielding 2-4x throughput improvements.

###  Basic Overlap Patterns
```cpp
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

###  Advanced Pipeline Processing
```cpp
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

##  Stream Synchronization Mechanisms

###  Comprehensive Synchronization Patterns
```cpp
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

##  Dynamic Stream Management

###  Adaptive Stream Allocation

See [AdaptiveStreamManager.h](../src/04_streams_concurrency/AdaptiveStreamManager.h) for the full implementation of the `AdaptiveStreamManager` class.
