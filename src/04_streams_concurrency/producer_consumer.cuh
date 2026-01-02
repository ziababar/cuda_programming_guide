#ifndef PRODUCER_CONSUMER_CUH
#define PRODUCER_CONSUMER_CUH

#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cstdio>

// Forward declarations
template<typename T>
void producer_worker(class StreamProducerConsumer<T>& system, int num_items);
template<typename T>
void consumer_worker(class StreamProducerConsumer<T>& system, int num_items);
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
            (float*)buffer, 1024, i);

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
            (float*)buffer, data_size / sizeof(T), sequence);

        // Release buffer
        system.release_consumer_buffer(buffer_id);

        // Simulate variable consumption rate
        std::this_thread::sleep_for(std::chrono::milliseconds(15 + (i % 15)));
    }

    printf("Consumer worker completed\n");
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

#endif // PRODUCER_CONSUMER_CUH
