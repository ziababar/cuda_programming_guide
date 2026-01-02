#ifndef BANDWIDTH_OPTIMIZER_CUH
#define BANDWIDTH_OPTIMIZER_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// Comprehensive bandwidth optimization techniques
class BandwidthOptimizer {
private:
    cudaDeviceProp device_props;
    float theoretical_bandwidth;

public:
    BandwidthOptimizer() {
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&device_props, device);

        // Calculate theoretical bandwidth
        theoretical_bandwidth = 2.0f * device_props.memoryClockRate *
                               (device_props.memoryBusWidth / 8) / 1.0e6;

        printf("=== Bandwidth Optimizer ===\n");
        printf("Device: %s\n", device_props.name);
        printf("Memory Clock Rate: %d kHz\n", device_props.memoryClockRate);
        printf("Memory Bus Width: %d bits\n", device_props.memoryBusWidth);
        printf("Theoretical Bandwidth: %.2f GB/s\n", theoretical_bandwidth);
        printf("===========================\n\n");
    }

    // Test different transfer sizes to find optimal chunk size
    void optimize_transfer_size() {
        printf("Optimizing Transfer Size:\n");

        const size_t max_size = 256 * 1024 * 1024; // 256MB
        const int num_iterations = 20;

        std::vector<size_t> test_sizes = {
            4 * 1024,           // 4KB
            64 * 1024,          // 64KB
            1024 * 1024,        // 1MB
            16 * 1024 * 1024,   // 16MB
            64 * 1024 * 1024,   // 64MB
            256 * 1024 * 1024   // 256MB
        };

        float *h_data, *d_data;
        cudaHostAlloc(&h_data, max_size, cudaHostAllocDefault);
        cudaMalloc(&d_data, max_size);

        // Initialize data
        for (size_t i = 0; i < max_size / sizeof(float); i++) {
            h_data[i] = i * 0.001f;
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        size_t optimal_size = 0;
        float best_bandwidth = 0.0f;

        for (size_t test_size : test_sizes) {
            // Measure H2D bandwidth
            cudaEventRecord(start);
            for (int i = 0; i < num_iterations; i++) {
                cudaMemcpyAsync(d_data, h_data, test_size,
                              cudaMemcpyHostToDevice, stream);
            }
            cudaStreamSynchronize(stream);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start, stop);
            float bandwidth = (test_size * num_iterations) /
                            (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

            printf("  Size: %8zu KB, Bandwidth: %6.2f GB/s (%.1f%% of theoretical)\n",
                   test_size / 1024, bandwidth,
                   (bandwidth / theoretical_bandwidth) * 100.0f);

            if (bandwidth > best_bandwidth) {
                best_bandwidth = bandwidth;
                optimal_size = test_size;
            }
        }

        printf("  Optimal transfer size: %zu KB (%.2f GB/s)\n\n",
               optimal_size / 1024, best_bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        cudaFreeHost(h_data);
        cudaFree(d_data);
    }

    // Test concurrent transfers with different stream counts
    void optimize_stream_count() {
        printf("Optimizing Stream Count for Concurrent Transfers:\n");

        const size_t transfer_size = 64 * 1024 * 1024; // 64MB per transfer
        const int max_streams = 8;

        for (int num_streams = 1; num_streams <= max_streams; num_streams++) {
            float total_bandwidth = test_concurrent_transfers(num_streams, transfer_size);
            printf("  %d streams: %.2f GB/s total bandwidth\n",
                   num_streams, total_bandwidth);
        }
        printf("\n");
    }

private:
    float test_concurrent_transfers(int num_streams, size_t transfer_size) {
        std::vector<float*> h_buffers(num_streams);
        std::vector<float*> d_buffers(num_streams);
        std::vector<cudaStream_t> streams(num_streams);

        // Allocate resources
        for (int i = 0; i < num_streams; i++) {
            cudaHostAlloc(&h_buffers[i], transfer_size, cudaHostAllocDefault);
            cudaMalloc(&d_buffers[i], transfer_size);
            cudaStreamCreate(&streams[i]);

            // Initialize data
            for (size_t j = 0; j < transfer_size / sizeof(float); j++) {
                h_buffers[i][j] = (i * 1000 + j) * 0.001f;
            }
        }

        // Measure concurrent transfers
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Launch all transfers concurrently
        for (int i = 0; i < num_streams; i++) {
            cudaMemcpyAsync(d_buffers[i], h_buffers[i], transfer_size,
                          cudaMemcpyHostToDevice, streams[i]);
        }

        // Wait for all transfers to complete
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        float total_bandwidth = (transfer_size * num_streams) /
                              (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        // Cleanup
        for (int i = 0; i < num_streams; i++) {
            cudaFreeHost(h_buffers[i]);
            cudaFree(d_buffers[i]);
            cudaStreamDestroy(streams[i]);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return total_bandwidth;
    }
};

#endif // BANDWIDTH_OPTIMIZER_CUH
