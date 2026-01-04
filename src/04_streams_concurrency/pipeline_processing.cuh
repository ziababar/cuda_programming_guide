#ifndef PIPELINE_PROCESSING_CUH
#define PIPELINE_PROCESSING_CUH

#include <cstdio>
#include <cmath>
#include <vector>

// Forward declarations
inline __global__ void stage1_kernel(float* input, float* output, int N);
inline __global__ void stage2_kernel(float* input, float* output, int N);

// Sophisticated pipeline with multiple processing stages
inline void pipeline_processing_demo(float* h_input, float* h_output,
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

inline __global__ void stage1_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = sqrt(fabs(input[tid]));
    }
}

inline __global__ void stage2_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = sin(input[tid]) + cos(input[tid]);
    }
}

#endif // PIPELINE_PROCESSING_CUH
