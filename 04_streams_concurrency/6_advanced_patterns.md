# Advanced Stream Patterns

Beyond basic stream operations, CUDA enables sophisticated coordination patterns that maximize GPU utilization through complex producer-consumer relationships, pipeline architectures, and dynamic load balancing strategies.

**[Back to Index](../README.md)** | **Previous: [CUDA Graphs](5_cuda_graphs.md)**

---

## **Producer-Consumer Patterns**

Producer-consumer patterns enable efficient data flow management where different components generate and consume data at potentially different rates, requiring sophisticated buffering and synchronization strategies.

**Source Code**: [`StreamProducerConsumer.h`](../../src/04_streams_concurrency/StreamProducerConsumer.h)

```cpp
#include "../src/04_streams_concurrency/StreamProducerConsumer.h"

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
```

## **Pipeline Architecture Patterns**

Stream-based pipelines enable complex multi-stage processing where each stage can operate independently and concurrently, maximizing GPU utilization and throughput.

**Source Code**: [`StreamPipeline.h`](../../src/04_streams_concurrency/StreamPipeline.h)

```cpp
#include "../src/04_streams_concurrency/StreamPipeline.h"
```

## **Dynamic Load Balancing**

Advanced stream patterns can dynamically distribute work across multiple streams based on real-time performance characteristics and system load.

**Source Code**: [`AdaptiveStreamBalancer.h`](../../src/04_streams_concurrency/AdaptiveStreamBalancer.h)

```cpp
#include "../src/04_streams_concurrency/AdaptiveStreamBalancer.h"

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
```

---

## **Nsight Debugging Tips**

- Use **Nsight Systems** to visualize:
  - Stream timelines
  - Overlap of memcopy and kernels
- Identify serialization caused by:
  - Shared resources
  - Host sync calls (`cudaDeviceSynchronize()`)
