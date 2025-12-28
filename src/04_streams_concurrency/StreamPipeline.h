#ifndef STREAM_PIPELINE_H
#define STREAM_PIPELINE_H

#include <vector>
#include <string>
#include <functional>
#include <cstdio>
#include <cuda_runtime.h>

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

        // Setup default functions must be done outside or via setter in this extracted version
        // to avoid dependency on specific kernels
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
            if (stage.process_func) {
                stage.process_func(stage.input_buffer, stage.output_buffer,
                                 stage.buffer_size, stage.stream);
            }

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
                if (stage.process_func) {
                    stage.process_func(stage.input_buffer, stage.output_buffer,
                                     stage.buffer_size, stage.stream);
                }

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

#endif // STREAM_PIPELINE_H
