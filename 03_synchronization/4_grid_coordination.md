#  Grid-Level Coordination

Grid-level coordination requires kernel boundaries or specialized synchronization primitives for coordination across thread blocks.

**Previous: [Warp-Level Synchronization](3_warp_synchronization.md)** | **Next: [Atomic Operations](5_atomic_operations.md)**

---

##  **Grid-Level Coordination**

###  **Multi-Kernel Coordination**

#### **Kernel Boundary Synchronization:**
```cpp
// Multi-stage algorithm using kernel boundaries
class MultiStageProcessor {
private:
    float* d_data;
    float* d_temp;
    int N;

public:
    MultiStageProcessor(int size) : N(size) {
        cudaMalloc(&d_data, N * sizeof(float));
        cudaMalloc(&d_temp, N * sizeof(float));
    }

    void process() {
        dim3 blocks((N + 255) / 256);
        dim3 threads(256);

        // Stage 1: Preprocessing
        preprocessing_kernel<<<blocks, threads>>>(d_data, d_temp, N);
        cudaDeviceSynchronize();  // Grid-level synchronization

        // Stage 2: Main computation
        main_computation_kernel<<<blocks, threads>>>(d_temp, d_data, N);
        cudaDeviceSynchronize();

        // Stage 3: Postprocessing
        postprocessing_kernel<<<blocks, threads>>>(d_data, d_temp, N);
        cudaDeviceSynchronize();
    }

    ~MultiStageProcessor() {
        cudaFree(d_data);
        cudaFree(d_temp);
    }
};

__global__ void preprocessing_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = sqrt(fabs(input[tid]));
    }
}

__global__ void main_computation_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float sum = 0.0f;
        for (int i = max(0, tid - 5); i <= min(N - 1, tid + 5); i++) {
            sum += input[i];
        }
        output[tid] = sum / 11.0f;  // Average of 11 elements
    }
}

__global__ void postprocessing_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = input[tid] * 2.0f + 1.0f;
    }
}
```

#### **Stream-Based Coordination:**
```cpp
// Coordinate multiple kernels using CUDA streams
void stream_based_coordination() {
    const int N = 1024 * 1024;
    const int num_streams = 4;

    float* h_data[num_streams];
    float* d_data[num_streams];
    cudaStream_t streams[num_streams];

    // Setup streams and data
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMallocHost(&h_data[i], N * sizeof(float));
        cudaMalloc(&d_data[i], N * sizeof(float));

        // Initialize data
        for (int j = 0; j < N; j++) {
            h_data[i][j] = i * N + j;
        }
    }

    // Launch kernels on different streams
    for (int i = 0; i < num_streams; i++) {
        // Async memory copy
        cudaMemcpyAsync(d_data[i], h_data[i], N * sizeof(float),
                       cudaMemcpyHostToDevice, streams[i]);

        // Kernel execution
        int blocks = (N + 255) / 256;
        stream_processing_kernel<<<blocks, 256, 0, streams[i]>>>(d_data[i], N);

        // Async copy back
        cudaMemcpyAsync(h_data[i], d_data[i], N * sizeof(float),
                       cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(h_data[i]);
        cudaFree(d_data[i]);
    }
}

__global__ void stream_processing_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = sin(data[tid]) + cos(data[tid]);
    }
}
```

###  **Inter-Block Communication**

#### **Global Memory Coordination:**
```cpp
// Inter-block communication using global memory
__global__ void inter_block_coordination(float* data, float* block_results,
                                        int* sync_counter, int N) {
    __shared__ float block_sum;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    // Phase 1: Each block computes local sum
    float thread_sum = 0.0f;
    for (int i = global_tid; i < N; i += gridDim.x * blockDim.x) {
        thread_sum += data[i];
    }

    // Block-level reduction
    __shared__ float shared_data[256];
    shared_data[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Store block result
    if (tid == 0) {
        block_sum = shared_data[0];
        block_results[bid] = block_sum;

        // Signal completion
        int count = atomicAdd(sync_counter, 1);

        // Last block computes final result
        if (count == gridDim.x - 1) {
            float total_sum = 0.0f;
            for (int i = 0; i < gridDim.x; i++) {
                total_sum += block_results[i];
            }

            printf("Total sum across all blocks: %.2f\n", total_sum);

            // Reset counter for next kernel launch
            *sync_counter = 0;
        }
    }
}
```
