#ifndef STREAM_SYNCHRONIZER_CUH
#define STREAM_SYNCHRONIZER_CUH

#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <cuda_runtime.h>

// Forward declaration of kernels
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

#endif // STREAM_SYNCHRONIZER_CUH
