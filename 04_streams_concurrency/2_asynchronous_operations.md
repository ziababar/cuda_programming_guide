# Asynchronous Operations

Asynchronous execution is the cornerstone of high-performance GPU programming, enabling overlapped computation, memory transfer concurrency, and sophisticated pipeline orchestration.

**[Back to Index](../README.md)** | **Previous: [Stream Fundamentals](1_stream_fundamentals.md)** | **Next: [Memory Transfer Optimization](3_memory_transfer.md)**

---

## **Compute-Transfer Overlap**

The ability to overlap computation with memory transfers is one of the most powerful features of CUDA streams, often yielding 2-4x throughput improvements.

### **Basic Overlap Patterns**

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

### **Advanced Pipeline Processing**

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

## **Dynamic Stream Management**

We use `AdaptiveStreamManager` for handling varying workloads dynamically.

**Source Code**: [`AdaptiveStreamManager.h`](../../src/04_streams_concurrency/AdaptiveStreamManager.h)

```cpp
#include "../src/04_streams_concurrency/AdaptiveStreamManager.h"
```
