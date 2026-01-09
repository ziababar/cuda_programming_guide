#ifndef STREAM_EXECUTION_CUH
#define STREAM_EXECUTION_CUH

#include <cstdio>
#include <cmath>

// Forward declarations of kernels
inline __global__ void preprocessing_kernel(float* data, int N);
inline __global__ void processing_kernel(float* data, int N);
inline __global__ void compute_intensive_kernel(float* data, int N);
inline __global__ void postprocessing_kernel(float* data, int N);
inline __global__ void initialization_kernel(float* data, int N);

// Demonstrate FIFO ordering and inter-stream concurrency
inline void demonstrate_stream_execution_model() {
    printf("=== Stream Execution Model ===\n");

    const int N = 1024 * 1024;
    float *h_data, *d_data1, *d_data2, *d_data3;

    // Allocate pinned memory for optimal transfer performance
    cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMalloc(&d_data3, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.001f;
    }

    // Create streams with different characteristics
    cudaStream_t stream_a, stream_b, stream_c;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);
    cudaStreamCreate(&stream_c);

    printf("1. FIFO Ordering Within Streams:\n");
    printf("   Operations within each stream execute in submission order\n");

    // Stream A: Sequential pipeline
    cudaMemcpyAsync(d_data1, h_data, N * sizeof(float),
                   cudaMemcpyHostToDevice, stream_a);         // Order: 1
    preprocessing_kernel<<<(N+255)/256, 256, 0, stream_a>>>(d_data1, N);    // Order: 2
    processing_kernel<<<(N+255)/256, 256, 0, stream_a>>>(d_data1, N);       // Order: 3
    cudaMemcpyAsync(h_data, d_data1, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_a);         // Order: 4

    printf("   Stream A: H2D -> Preprocess -> Process -> D2H\n");

    // Stream B: Different pipeline
    cudaMemcpyAsync(d_data2, h_data, N * sizeof(float),
                   cudaMemcpyHostToDevice, stream_b);         // Concurrent with Stream A
    compute_intensive_kernel<<<(N+255)/256, 256, 0, stream_b>>>(d_data2, N); // Different work
    postprocessing_kernel<<<(N+255)/256, 256, 0, stream_b>>>(d_data2, N);

    printf("   Stream B: H2D -> Intensive -> Postprocess (concurrent)\n");

    // Stream C: Memory operations
    cudaMemsetAsync(d_data3, 0, N * sizeof(float), stream_c);              // Concurrent init
    initialization_kernel<<<(N+255)/256, 256, 0, stream_c>>>(d_data3, N);

    printf("   Stream C: Memset -> Initialize (concurrent)\n\n");

    printf("2. Inter-Stream Concurrency:\n");
    printf("   Different streams can execute concurrently\n");
    printf("   GPU scheduler interleaves stream operations\n");
    printf("   Actual concurrency depends on resource availability\n\n");

    // Wait for all streams to complete
    cudaStreamSynchronize(stream_a);
    cudaStreamSynchronize(stream_b);
    cudaStreamSynchronize(stream_c);

    printf("3. Synchronization Points:\n");
    printf("   cudaStreamSynchronize() - Wait for specific stream\n");
    printf("   cudaDeviceSynchronize() - Wait for all streams\n");
    printf("   Events - Fine-grained inter-stream dependencies\n");

    // Cleanup
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    cudaStreamDestroy(stream_c);
    cudaFreeHost(h_data);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
}

inline __global__ void preprocessing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = sqrt(fabs(data[tid]));
    }
}

inline __global__ void processing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = sin(data[tid]) + cos(data[tid]);
    }
}

inline __global__ void compute_intensive_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float value = data[tid];
        // Intensive computation
        for (int i = 0; i < 100; i++) {
            value = sin(value) + cos(value);
        }
        data[tid] = value;
    }
}

inline __global__ void postprocessing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

inline __global__ void initialization_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = tid * 0.01f;
    }
}

#endif // STREAM_EXECUTION_CUH
