#pragma once
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <iostream>
#include <cmath>

// Use the cuda::pipeline namespace
using namespace cuda;

// Kernel using cp.async with cuda::pipeline
// Computes a simple transformation: output[i] = input[i] * 2.0f
// This demonstrates the pipeline pattern:
// 1. Issue async copy for next tile
// 2. Compute on current tile
// 3. Wait for next tile
template<int STAGES=3>
__global__ void async_copy_pipeline_kernel(float* input, float* output, size_t n) {
    extern __shared__ float shared_mem[];
    float* smem = shared_mem;

    // Pipeline object
    pipeline<thread_scope_thread> pipe = make_pipeline();

    // Calculate offsets
    size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * blockDim.x;
    size_t tile_size = blockDim.x; // One element per thread per tile

    // Pointers for ring buffer in shared memory
    // Each stage needs 'tile_size' float elements
    // Total shared memory needed: STAGES * tile_size * sizeof(float)

    // Prologue: Fill the pipeline
    // We issue copies for the first (STAGES - 1) batches
    // Note: We need to handle boundary conditions carefully

    // Total number of batches per grid stride loop
    size_t num_batches = (n + grid_stride - 1) / grid_stride;

    // Loop over all batches
    for (size_t i = 0; i < num_batches; ++i) {
        // Stage index for the batch we are about to ISSUE
        int next_stage = (i + STAGES - 1) % STAGES;

        // 1. Issue copy for a future batch (prologue + steady state)
        // We want to keep STAGES-1 copies in flight
        // The first few iterations fill the pipe
        if (i < num_batches) { // Simple loop structure
             size_t fetch_idx = i + STAGES - 1;
             if (fetch_idx < num_batches) {
                 // Calculate which stage buffer to use for this fetch
                 int fetch_stage = fetch_idx % STAGES;
                 size_t fetch_offset = global_tid + fetch_idx * grid_stride;

                 if (fetch_offset < n) {
                     pipe.producer_acquire();
                     cuda::memcpy_async(&smem[fetch_stage * tile_size + threadIdx.x],
                                      &input[fetch_offset],
                                      sizeof(float),
                                      pipe);
                     pipe.producer_commit();
                 }
             }
        }

        // Wait for the oldest batch to arrive if we have filled the pipe
        // or if we are consuming
        // Actually, let's use a simpler canonical loop structure provided in NVIDIA docs
    }
}

// Rewriting kernel for clarity using standard loop pattern
template<int STAGES=3>
__global__ void async_copy_pipeline_v2(float* input, float* output, size_t n) {
    extern __shared__ float shared_mem[];
    float* smem = shared_mem;

    pipeline<thread_scope_thread> pipe = make_pipeline();

    size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * blockDim.x;
    size_t tile_size = blockDim.x;

    // Prologue: Fill pipeline
    for (int s = 0; s < STAGES; ++s) {
        pipe.producer_acquire();
        size_t offset = global_tid + s * grid_stride;
        if (offset < n) {
            cuda::memcpy_async(&smem[s * tile_size + threadIdx.x],
                             &input[offset], sizeof(float), pipe);
        }
        pipe.producer_commit();
    }

    // Main loop
    // Note: This simple loop assumes n is large enough or handled by checks
    size_t i = 0;
    for (; i < (n + grid_stride - 1) / grid_stride; ++i) {
        // Consumer: Process stage 'i % STAGES'
        pipe.consumer_wait();

        size_t curr_offset = global_tid + i * grid_stride;
        int curr_stage = i % STAGES;

        if (curr_offset < n) {
            float val = smem[curr_stage * tile_size + threadIdx.x];
            output[curr_offset] = val * 2.0f;
        }
        pipe.consumer_release();

        // Producer: Issue next stage
        // The slot 'curr_stage' is now free to be overwritten by 'i + STAGES'
        pipe.producer_acquire();
        size_t next_offset = global_tid + (i + STAGES) * grid_stride;
        if (next_offset < n) {
             cuda::memcpy_async(&smem[curr_stage * tile_size + threadIdx.x],
                              &input[next_offset], sizeof(float), pipe);
        }
        pipe.producer_commit();
    }

    // Epilogue: Not strictly needed if loop condition handles it,
    // but we issued STAGES extra commits at the end that are out of bounds
    // (handled by next_offset < n check which does nothing but empty commit)
}

class AsyncCopyDemo {
public:
    void run_demo() {
        std::cout << "=== Asynchronous Copy (cp.async) Demo ===" << std::endl;

        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        if (prop.major < 8) {
            std::cout << "Skipping: cp.async requires Compute Capability 8.0+ (Ampere)" << std::endl;
            std::cout << "Current device: " << prop.name << " (CC " << prop.major << "." << prop.minor << ")" << std::endl;
            return;
        }

        const size_t N = 1024 * 1024;
        const size_t bytes = N * sizeof(float);

        float *h_in, *h_out, *d_in, *d_out;
        cudaMallocHost(&h_in, bytes);
        cudaMallocHost(&h_out, bytes);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        for(size_t i=0; i<N; ++i) h_in[i] = 1.0f;

        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        const int threads = 128;
        const int blocks = (N + threads - 1) / threads;
        // 3 stages * 128 floats * 4 bytes
        const int smem_size = 3 * threads * sizeof(float);

        std::cout << "Launching kernel with " << blocks << " blocks, " << threads << " threads." << std::endl;

        async_copy_pipeline_v2<3><<<blocks, threads, smem_size>>>(d_in, d_out, N);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "Kernel completed successfully." << std::endl;
        }

        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for(size_t i=0; i<N; ++i) {
            if (abs(h_out[i] - 2.0f) > 1e-5) {
                correct = false;
                std::cout << "Mismatch at " << i << ": " << h_out[i] << " != 2.0" << std::endl;
                break;
            }
        }

        if(correct) std::cout << "Verification PASSED!" << std::endl;
        else std::cout << "Verification FAILED!" << std::endl;

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    }
};
