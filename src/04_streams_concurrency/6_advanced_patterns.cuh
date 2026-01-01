#ifndef ADVANCED_PATTERNS_H
#define ADVANCED_PATTERNS_H

#include <cstdio>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>

// Forward declarations
__global__ void producer_kernel(float* output, int N, int item_id);
__global__ void consumer_kernel(float* input, int N, int sequence);
__global__ void pipeline_preprocess_kernel(float* input, float* output, int N);
__global__ void pipeline_compute_kernel(float* input, float* output, int N);
__global__ void pipeline_postprocess_kernel(float* input, float* output, int N);
__global__ void pipeline_generic_kernel(float* input, float* output, int N, int stage_id);
__global__ void light_compute_kernel(int task_id);
__global__ void medium_compute_kernel(int task_id);
__global__ void heavy_compute_kernel(int task_id);

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

#endif // ADVANCED_PATTERNS_H
