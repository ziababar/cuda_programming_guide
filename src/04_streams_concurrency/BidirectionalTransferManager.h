#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>

// Forward declaration for kernel
__global__ void bidirectional_compute_kernel(float* data, int N, int iteration);

// Sophisticated bidirectional transfer patterns
class BidirectionalTransferManager {
private:
    struct TransferBuffer {
        float* h_buffer;
        float* d_buffer;
        size_t size;
        cudaStream_t upload_stream;
        cudaStream_t download_stream;
        cudaEvent_t upload_complete;
        cudaEvent_t download_complete;
    };

    std::vector<TransferBuffer> buffers;
    int num_buffers;

public:
    BidirectionalTransferManager(int buffer_count, size_t buffer_size)
        : num_buffers(buffer_count) {

        buffers.resize(buffer_count);

        for (int i = 0; i < buffer_count; i++) {
            // Allocate pinned host memory
            cudaHostAlloc(&buffers[i].h_buffer, buffer_size, cudaHostAllocDefault);

            // Allocate device memory
            cudaMalloc(&buffers[i].d_buffer, buffer_size);

            // Create dedicated streams for each direction
            cudaStreamCreate(&buffers[i].upload_stream);
            cudaStreamCreate(&buffers[i].download_stream);

            // Create events for synchronization
            cudaEventCreate(&buffers[i].upload_complete);
            cudaEventCreate(&buffers[i].download_complete);

            buffers[i].size = buffer_size;

            // Initialize with test data
            for (size_t j = 0; j < buffer_size / sizeof(float); j++) {
                buffers[i].h_buffer[j] = (i * 10000 + j) * 0.0001f;
            }
        }

        printf("BidirectionalTransferManager initialized with %d buffers\n", buffer_count);
    }

    // Demonstrate overlapped bidirectional transfers
    void demonstrate_bidirectional_overlap() {
        printf("=== Bidirectional Transfer Overlap ===\n");

        cudaEvent_t overall_start, overall_stop;
        cudaEventCreate(&overall_start);
        cudaEventCreate(&overall_stop);

        // Method 1: Sequential transfers
        printf("1. Sequential bidirectional transfers:\n");
        cudaEventRecord(overall_start);

        for (int i = 0; i < num_buffers; i++) {
            // Upload then download sequentially
            cudaMemcpy(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                      cudaMemcpyHostToDevice);
            cudaMemcpy(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                      cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(overall_stop);
        cudaEventSynchronize(overall_stop);

        float sequential_time;
        cudaEventElapsedTime(&sequential_time, overall_start, overall_stop);
        printf("   Sequential time: %.2f ms\n", sequential_time);

        // Method 2: Overlapped transfers using streams
        printf("2. Overlapped bidirectional transfers:\n");
        cudaEventRecord(overall_start);

        // Launch all uploads first
        for (int i = 0; i < num_buffers; i++) {
            cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                          cudaMemcpyHostToDevice, buffers[i].upload_stream);
            cudaEventRecord(buffers[i].upload_complete, buffers[i].upload_stream);
        }

        // Launch downloads that depend on uploads
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamWaitEvent(buffers[i].download_stream, buffers[i].upload_complete, 0);
            cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                          cudaMemcpyDeviceToHost, buffers[i].download_stream);
            cudaEventRecord(buffers[i].download_complete, buffers[i].download_stream);
        }

        // Wait for all downloads to complete
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamSynchronize(buffers[i].download_stream);
        }

        cudaEventRecord(overall_stop);
        cudaEventSynchronize(overall_stop);

        float overlapped_time;
        cudaEventElapsedTime(&overlapped_time, overall_start, overall_stop);
        printf("   Overlapped time: %.2f ms\n", overlapped_time);
        printf("   Speedup: %.2fx\n", sequential_time / overlapped_time);

        // Method 3: Advanced pipeline with computation
        printf("3. Pipeline with computation overlap:\n");
        pipeline_with_computation();

        cudaEventDestroy(overall_start);
        cudaEventDestroy(overall_stop);
    }

    void pipeline_with_computation() {
        cudaEvent_t pipeline_start, pipeline_stop;
        cudaEventCreate(&pipeline_start);
        cudaEventCreate(&pipeline_stop);

        cudaEventRecord(pipeline_start);

        // Complex pipeline: Upload -> Compute -> Download
        for (int iteration = 0; iteration < 3; iteration++) {
            for (int i = 0; i < num_buffers; i++) {
                // Stage 1: Upload data
                cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                              cudaMemcpyHostToDevice, buffers[i].upload_stream);
                cudaEventRecord(buffers[i].upload_complete, buffers[i].upload_stream);

                // Stage 2: Compute (depends on upload)
                cudaStreamWaitEvent(buffers[i].download_stream, buffers[i].upload_complete, 0);

                int num_elements = buffers[i].size / sizeof(float);
                bidirectional_compute_kernel<<<(num_elements + 255)/256, 256, 0,
                                             buffers[i].download_stream>>>(
                    buffers[i].d_buffer, num_elements, iteration);

                // Stage 3: Download result
                cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                              cudaMemcpyDeviceToHost, buffers[i].download_stream);
                cudaEventRecord(buffers[i].download_complete, buffers[i].download_stream);
            }

            // Wait for this iteration to complete before starting next
            for (int i = 0; i < num_buffers; i++) {
                cudaStreamSynchronize(buffers[i].download_stream);
            }

            printf("   Pipeline iteration %d complete\n", iteration + 1);
        }

        cudaEventRecord(pipeline_stop);
        cudaEventSynchronize(pipeline_stop);

        float pipeline_time;
        cudaEventElapsedTime(&pipeline_time, pipeline_start, pipeline_stop);
        printf("   Total pipeline time: %.2f ms\n", pipeline_time);

        cudaEventDestroy(pipeline_start);
        cudaEventDestroy(pipeline_stop);
    }

    // Analyze transfer bandwidth utilization
    void analyze_bandwidth_utilization() {
        printf("=== Bandwidth Utilization Analysis ===\n");

        const int num_measurements = 10;
        float total_bandwidth = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int measurement = 0; measurement < num_measurements; measurement++) {
            cudaEventRecord(start);

            // Concurrent bidirectional transfers
            for (int i = 0; i < num_buffers; i++) {
                cudaMemcpyAsync(buffers[i].d_buffer, buffers[i].h_buffer, buffers[i].size,
                              cudaMemcpyHostToDevice, buffers[i].upload_stream);
                cudaMemcpyAsync(buffers[i].h_buffer, buffers[i].d_buffer, buffers[i].size,
                              cudaMemcpyDeviceToHost, buffers[i].download_stream);
            }

            // Synchronize all transfers
            for (int i = 0; i < num_buffers; i++) {
                cudaStreamSynchronize(buffers[i].upload_stream);
                cudaStreamSynchronize(buffers[i].download_stream);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start, stop);

            // Calculate total bandwidth (upload + download)
            size_t total_bytes = buffers[0].size * num_buffers * 2; // *2 for bidirectional
            float bandwidth = total_bytes / (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            total_bandwidth += bandwidth;

            printf("   Measurement %d: %.2f GB/s\n", measurement + 1, bandwidth);
        }

        float avg_bandwidth = total_bandwidth / num_measurements;
        printf("   Average bandwidth: %.2f GB/s\n", avg_bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    ~BidirectionalTransferManager() {
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamSynchronize(buffers[i].upload_stream);
            cudaStreamSynchronize(buffers[i].download_stream);

            cudaFreeHost(buffers[i].h_buffer);
            cudaFree(buffers[i].d_buffer);
            cudaStreamDestroy(buffers[i].upload_stream);
            cudaStreamDestroy(buffers[i].download_stream);
            cudaEventDestroy(buffers[i].upload_complete);
            cudaEventDestroy(buffers[i].download_complete);
        }

        printf("BidirectionalTransferManager cleanup complete\n");
    }
};

__global__ void bidirectional_compute_kernel(float* data, int N, int iteration) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Different computation per iteration
        for (int i = 0; i < (iteration + 1) * 50; i++) {
            value = sin(value) + cos(value * 0.1f);
        }

        data[tid] = value;
    }
}
