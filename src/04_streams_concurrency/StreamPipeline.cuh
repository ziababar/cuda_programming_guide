#ifndef STREAM_PIPELINE_CUH
#define STREAM_PIPELINE_CUH

#include <vector>
#include <string>
#include <functional>
#include <cstdio>

// Forward declarations
inline __global__ void pipeline_preprocess_kernel(float* input, float* output, int N);
inline __global__ void pipeline_compute_kernel(float* input, float* output, int N);
inline __global__ void pipeline_postprocess_kernel(float* input, float* output, int N);
inline __global__ void pipeline_generic_kernel(float* input, float* output, int N, int stage_id);

// Sophisticated multi-stage pipeline with dynamic load balancing
class StreamPipeline {
private:
    struct PipelineStage {
        std::string name;
        cudaStream_t stream;
        std::function<void(float*, float*, int, cudaStream_t)> process_func;
        // In overlapped mode, we need buffers per batch
        // We'll manage buffers dynamically or use a pool
        int buffer_size;
        float avg_processing_time;
        int completed_batches;
        cudaEvent_t stage_complete;
    };

    std::vector<PipelineStage> stages;
    // Pool of intermediate buffers: [batch_id][stage_id] -> buffer
    std::vector<std::vector<float*>> batch_intermediate_buffers;
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

        // Initialize pipeline stages
        for (int i = 0; i < num_stages; i++) {
            PipelineStage& stage = stages[i];
            stage.name = "Stage_" + std::to_string(i);

            cudaStreamCreate(&stage.stream);
            cudaEventCreate(&stage.stage_complete);

            stage.buffer_size = buffer_elements;
            stage.avg_processing_time = 0.0f;
            stage.completed_batches = 0;

            printf("Initialized %s\n", stage.name.c_str());
        }

        setup_default_processing_functions();
        is_initialized = true;

        printf("StreamPipeline initialization complete\n");
    }

    // Allocate buffers for a batch (input + intermediates + output for last stage)
    // Actually, pipeline usually transforms input -> buffer1 -> buffer2 -> ... -> output
    // We need num_stages + 1 buffers if we consider input/output external,
    // or num_stages - 1 intermediate buffers.
    // Let's stick to the previous model: intermediate_buffers was size num_stages+1
    // where 0 is input copy dest, and num_stages is output copy src.
    void allocate_batch_buffers(int num_batches) {
        batch_intermediate_buffers.resize(num_batches);
        for(int b = 0; b < num_batches; ++b) {
            batch_intermediate_buffers[b].resize(num_stages + 1);
            for(int i = 0; i <= num_stages; ++i) {
                cudaMalloc(&batch_intermediate_buffers[b][i], buffer_elements * sizeof(float));
            }
        }
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

    // Execute pipeline on input data (Single batch mode)
    void execute_pipeline(float* input_data, float* output_data,
                         bool measure_performance = true) {
        if (!is_initialized) {
            printf("Pipeline not initialized\n");
            return;
        }

        // For single execution, use a temporary set of buffers or check if we have any
        if (batch_intermediate_buffers.empty()) {
            allocate_batch_buffers(1);
        }

        auto& buffers = batch_intermediate_buffers[0];

        printf("=== Executing Pipeline ===\n");

        // Copy input data to first buffer
        cudaMemcpy(buffers[0], input_data,
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

            // Execute stage processing using the specific buffers for this run
            stage.process_func(buffers[i], buffers[i+1],
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
        cudaMemcpy(output_data, buffers[num_stages],
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

        // Ensure we have enough buffer sets for the batches
        if (batch_intermediate_buffers.size() < num_batches) {
             // Free existing if any to resize cleanly, or just append
             // For simplicity, reallocate all to match needed batches (demo code)
             for(auto& vec : batch_intermediate_buffers) {
                 for(auto* ptr : vec) cudaFree(ptr);
             }
             batch_intermediate_buffers.clear();
             allocate_batch_buffers(num_batches);
        }

        // Overlapped execution for maximum throughput
        cudaEvent_t batch_start, batch_end;
        cudaEventCreate(&batch_start);
        cudaEventCreate(&batch_end);

        cudaEventRecord(batch_start);

        for (int batch = 0; batch < num_batches; batch++) {
            printf("Starting batch %d/%d\n", batch + 1, num_batches);

            auto& buffers = batch_intermediate_buffers[batch];

            // Copy input data asynchronously
            cudaMemcpyAsync(buffers[0], input_batches[batch],
                           buffer_elements * sizeof(float),
                           cudaMemcpyHostToDevice, stages[0].stream);

            // Execute pipeline stages with proper dependencies
            for (int i = 0; i < num_stages; i++) {
                PipelineStage& stage = stages[i];

                // Wait for previous stage OF THIS BATCH
                // Note: In a pipeline, Stage[i] of Batch[k] depends on Stage[i-1] of Batch[k].
                // But Stage[i] also effectively shares the compute resource (stream) with Stage[i] of Batch[k-1].
                // The stream FIFO order handles resource serialization.
                // We mainly need to ensure data dependency: Stage[i] waits for input buffer (buffers[i]) to be ready.
                // buffers[i] is produced by Stage[i-1].

                if (i > 0) {
                    // This wait is tricky. stages[i-1].stage_complete tracks the completion of Stage[i-1] for the *previous* command issued to it.
                    // But here we are issuing commands in a loop.
                    // Ideally, we record an event for *this specific batch's* previous stage completion.
                    // However, since everything for one batch is issued in sequence, and streams are separate:
                    // If Stage[i-1] runs on Stream[i-1], and Stage[i] runs on Stream[i].
                    // We need Stream[i] to wait for Stream[i-1] to finish writing to buffers[i].

                    // We need an event specific to this batch's stage completion.
                    // Or, we can use the stream semantics:
                    // Since we are issuing batch 0, then batch 1...
                    // The 'stage_complete' event in the struct is reused. This is dangerous if loop is tight.
                    // BUT: We issue all stages for Batch K. Then all stages for Batch K+1.
                    // When we are at Batch K, Stage i:
                    // We record an event on Stream i-1 (after Stage i-1 work).
                    // We wait on Stream i for that event.

                    // To do this correctly without per-batch events, we need to create temp events or manage them.
                    // For this demo, let's create a temporary event for dependency.

                    cudaEvent_t dependency_event;
                    cudaEventCreate(&dependency_event);
                    cudaEventRecord(dependency_event, stages[i-1].stream);
                    cudaStreamWaitEvent(stage.stream, dependency_event, 0);
                    cudaEventDestroy(dependency_event);
                }

                // Execute processing
                stage.process_func(buffers[i], buffers[i+1],
                                 stage.buffer_size, stage.stream);

                // Record completion (global stage progress)
                cudaEventRecord(stage.stage_complete, stage.stream);
            }

            // Copy output data asynchronously
            cudaMemcpyAsync(output_batches[batch], buffers[num_stages],
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
        for (auto& vec : batch_intermediate_buffers) {
            for (auto* ptr : vec) {
                cudaFree(ptr);
            }
        }

        printf("StreamPipeline cleanup complete\n");
    }
};

// Demonstrate advanced pipeline patterns
inline void demonstrate_pipeline_patterns() {
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
inline __global__ void pipeline_preprocess_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Normalization and basic preprocessing
        output[tid] = (input[tid] - 128.0f) / 255.0f;
    }
}

inline __global__ void pipeline_compute_kernel(float* input, float* output, int N) {
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

inline __global__ void pipeline_postprocess_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Post-processing - scaling and clamping
        float value = input[tid] * 255.0f + 128.0f;
        output[tid] = fmaxf(0.0f, fminf(255.0f, value));
    }
}

inline __global__ void pipeline_generic_kernel(float* input, float* output, int N, int stage_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Generic processing based on stage ID
        output[tid] = input[tid] * (stage_id + 1) + 0.1f;
    }
}

#endif // STREAM_PIPELINE_CUH
