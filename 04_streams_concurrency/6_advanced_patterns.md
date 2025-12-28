#  Advanced Stream Patterns

Beyond basic stream operations, CUDA enables sophisticated coordination patterns that maximize GPU utilization through complex producer-consumer relationships, pipeline architectures, and dynamic load balancing strategies.

##  Producer-Consumer Patterns

Producer-consumer patterns enable efficient data flow management where different components generate and consume data at potentially different rates, requiring sophisticated buffering and synchronization strategies.

###  Multi-Buffer Producer-Consumer System
See [StreamProducerConsumer.h](../src/04_streams_concurrency/StreamProducerConsumer.h) for the full implementation of the `StreamProducerConsumer` class.

```cpp
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

##  Pipeline Architecture Patterns

Stream-based pipelines enable complex multi-stage processing where each stage can operate independently and concurrently, maximizing GPU utilization and throughput.

###  Multi-Stage Processing Pipeline
See [StreamPipeline.h](../src/04_streams_concurrency/StreamPipeline.h) for the full implementation of the `StreamPipeline` class.

```cpp
// Demonstrate advanced pipeline patterns
void demonstrate_pipeline_patterns() {
    printf("=== Pipeline Patterns Demonstration ===\n");

    const int buffer_size = 1024 * 1024; // 1M elements
    const int num_batches = 5;

    // Create pipeline with 4 stages
    StreamPipeline pipeline(4, buffer_size);

    // Configure pipeline stages with specific kernels
    pipeline.set_stage_processor(0, [](float* input, float* output, int N, cudaStream_t stream) {
        pipeline_preprocess_kernel<<<(N+255)/256, 256, 0, stream>>>(input, output, N);
    }, "Preprocessing");

    pipeline.set_stage_processor(1, [](float* input, float* output, int N, cudaStream_t stream) {
        pipeline_compute_kernel<<<(N+255)/256, 256, 0, stream>>>(input, output, N);
    }, "MainCompute");

    pipeline.set_stage_processor(2, [](float* input, float* output, int N, cudaStream_t stream) {
        pipeline_postprocess_kernel<<<(N+255)/256, 256, 0, stream>>>(input, output, N);
    }, "Postprocessing");

    pipeline.set_stage_processor(3, [](float* input, float* output, int N, cudaStream_t stream) {
        pipeline_generic_kernel<<<(N+255)/256, 256, 0, stream>>>(input, output, N, 3);
    }, "Finalize");

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

##  Dynamic Load Balancing

Advanced stream patterns can dynamically distribute work across multiple streams based on real-time performance characteristics and system load.

###  Adaptive Stream Load Balancer
See [AdaptiveStreamBalancer.h](../src/04_streams_concurrency/AdaptiveStreamBalancer.h) for the full implementation of the `AdaptiveStreamBalancer` class.

```cpp
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

##  Nsight Debugging Tips

- Use **Nsight Systems** to visualize:
  - Stream timelines
  - Overlap of memcopy and kernels
- Identify serialization caused by:
  - Shared resources
  - Host sync calls (`cudaDeviceSynchronize()`)
