#  Advanced Patterns

Advanced synchronization patterns enable complex algorithms and high-performance coordination.

**Previous: [Debugging Synchronization](7_synchronization_debugging.md)** | **Next: [Execution Constraints Guide](../01_execution_model/5_execution_constraints_guide.md)**

##  **Advanced Patterns**

###  **Wave Synchronization**

#### **Multi-Block Wave Processing:**
```cpp
// Implement wave-style synchronization across blocks
__global__ void wave_synchronization(float* data, int* wave_counter,
                                    int wave_size, int N) {
    __shared__ float block_result;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Phase 1: Local computation
    float local_sum = 0.0f;
    for (int i = bid * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) {
        local_sum += data[i];
    }

    // Block-level reduction
    __shared__ float shared_sums[256];
    shared_sums[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_result = shared_sums[0];

        // Wave synchronization
        int wave_id = bid / wave_size;
        int pos_in_wave = bid % wave_size;

        // Increment wave counter
        int count = atomicAdd(&wave_counter[wave_id], 1);

        // Last block in wave processes results
        if (count == wave_size - 1) {
            printf("Wave %d completed processing\n", wave_id);
            // Reset counter for next iteration
            wave_counter[wave_id] = 0;
        }
    }
}
```

#### **Pipeline Synchronization:**
```cpp
// Multi-stage pipeline with stage synchronization
__global__ void pipeline_stage_kernel(float* input_buffer, float* output_buffer,
                                     int* stage_counters, int stage_id,
                                     int pipeline_depth, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Wait for previous stage to complete
        if (stage_id > 0) {
            while (atomicAdd(&stage_counters[stage_id - 1], 0) < N) {
                // Busy wait for previous stage
            }
        }

        // Process data for this stage
        float data = input_buffer[tid];

        // Stage-specific processing
        switch (stage_id) {
            case 0: data = sqrt(fabs(data)); break;
            case 1: data = sin(data); break;
            case 2: data = data * 2.0f + 1.0f; break;
            default: data = exp(-data); break;
        }

        output_buffer[tid] = data;

        // Signal completion of this thread's work
        atomicAdd(&stage_counters[stage_id], 1);
    }
}

// Host code to run pipeline
void run_pipeline(float* h_data, int N) {
    float *d_buffer1, *d_buffer2;
    int *d_stage_counters;
    const int pipeline_depth = 4;

    cudaMalloc(&d_buffer1, N * sizeof(float));
    cudaMalloc(&d_buffer2, N * sizeof(float));
    cudaMalloc(&d_stage_counters, pipeline_depth * sizeof(int));

    // Initialize
    cudaMemcpy(d_buffer1, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_stage_counters, 0, pipeline_depth * sizeof(int));

    dim3 blocks((N + 255) / 256);
    dim3 threads(256);

    // Launch pipeline stages
    for (int stage = 0; stage < pipeline_depth; stage++) {
        float* input = (stage % 2 == 0) ? d_buffer1 : d_buffer2;
        float* output = (stage % 2 == 0) ? d_buffer2 : d_buffer1;

        pipeline_stage_kernel<<<blocks, threads>>>(input, output, d_stage_counters,
                                                  stage, pipeline_depth, N);
    }

    cudaDeviceSynchronize();

    // Copy final result back
    float* final_buffer = (pipeline_depth % 2 == 0) ? d_buffer1 : d_buffer2;
    cudaMemcpy(h_data, final_buffer, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_buffer1);
    cudaFree(d_buffer2);
    cudaFree(d_stage_counters);
}
```

---

##  **Key Takeaways**

1. ** Understand Scope**: Choose the right synchronization level for your coordination needs
2. ** Block Barriers**: Use `__syncthreads()` for shared memory coordination within blocks
3. ** Warp Primitives**: Leverage SIMT properties and warp-level primitives for efficient coordination
4. ** Atomic Operations**: Use atomics judiciously - reduce contention through aggregation
5. ** Memory Consistency**: Understand memory fences and consistency models for correct behavior
6. ** Debug Systematically**: Use systematic testing to identify and fix synchronization bugs

##  **Related Guides**

- **Next Step**: [ Execution Constraints Guide](../01_execution_model/5_execution_constraints_guide.md) - Resource limits and workarounds
- **Foundation**: [ Streaming Multiprocessors Guide](../01_execution_model/4_streaming_multiprocessors_deep.md) - SM architecture and scheduling
- **Parallelism**: [ Warp Execution Guide](../01_execution_model/3_warp_execution.md) - Warp-level coordination
- **Overview**: [ Execution Model Overview](../01_execution_model/1_cuda_execution_model.md) - Quick reference and navigation

---

** Pro Tip**: Synchronization is about coordination, not just correctness. Choose the right synchronization primitive for both correctness AND performance!
