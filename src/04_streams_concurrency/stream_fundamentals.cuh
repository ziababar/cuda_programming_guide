#ifndef STREAM_FUNDAMENTALS_CUH
#define STREAM_FUNDAMENTALS_CUH

#include <cstdio>
#include <cmath>
#include <chrono>

// Forward declaration of kernel
inline __global__ void simple_kernel(float* data, int N);

// Comprehensive stream type demonstration
inline void demonstrate_stream_fundamentals() {
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

inline __global__ void simple_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Simple computation to demonstrate stream behavior
        data[tid] = tid * 2.0f + sin(tid * 0.01f);
    }
}

#endif // STREAM_FUNDAMENTALS_CUH
