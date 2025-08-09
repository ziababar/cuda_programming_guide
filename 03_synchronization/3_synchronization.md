#  Synchronization Complete Guide

Synchronization is the cornerstone of parallel programming in CUDA. Understanding how threads coordinate, share data safely, and maintain consistency is essential for writing correct and efficient GPU applications.

**[Back to Execution Model](../01_execution_model/1_cuda_execution_model.md)** | **Previous: [Streaming Multiprocessors Guide](../01_execution_model/1c_streaming_multiprocessors_deep.md)** | **Next: [Memory Hierarchy](../02_memory_hierarchy/2_cuda_memory_hierarchy.md)**

---

##  **Table of Contents**

1. [ Synchronization Fundamentals](#-synchronization-fundamentals)
2. [ Block-Level Synchronization](#-block-level-synchronization)
3. [ Warp-Level Synchronization](#-warp-level-synchronization)
4. [ Grid-Level Coordination](#-grid-level-coordination)
5. [ Atomic Operations](#-atomic-operations)
6. [ Memory Consistency](#-memory-consistency)
7. [ Debugging Synchronization](#-debugging-synchronization)
8. [ Advanced Patterns](#-advanced-patterns)

---

##  **Synchronization Fundamentals**

CUDA provides multiple levels of synchronization, each with different scopes, performance characteristics, and use cases.

###  **Synchronization Hierarchy**

| Level | Scope | Mechanism | Performance | Use Cases |
|-------|-------|-----------|-------------|-----------|
| **Thread** | Individual thread | None required | N/A | Independent operations |
| **Warp** | 32 threads | Implicit SIMT + `__syncwarp()` | Very fast | Warp-level algorithms |
| **Block** | Thread block | `__syncthreads()` | Fast | Shared memory cooperation |
| **Grid** | Entire kernel | Kernel boundaries | Slow | Multi-stage algorithms |
| **Device** | Multiple kernels | `cudaDeviceSynchronize()` | Very slow | Host-device coordination |

###  **Synchronization Models**

#### **Barrier Synchronization:**
```cpp
// Basic barrier synchronization patterns
__global__ void barrier_synchronization_demo() {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;

    // Phase 1: All threads write to shared memory
    shared_data[tid] = tid * 2.0f;

    // Barrier: Ensure all writes complete before reads
    __syncthreads();

    // Phase 2: All threads can safely read any shared memory location
    float neighbor_value = shared_data[(tid + 1) % blockDim.x];
    shared_data[tid] = shared_data[tid] + neighbor_value;

    // Another barrier before final phase
    __syncthreads();

    // Phase 3: Final processing with consistent shared memory state
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[i];
        }
        printf("Block sum: %.2f\n", sum);
    }
}
```

#### **Producer-Consumer Pattern:**
```cpp
// Producer-consumer synchronization with shared memory
__global__ void producer_consumer_demo() {
    __shared__ float buffer[128];
    __shared__ int write_pos;
    __shared__ int read_pos;
    __shared__ int data_count;

    int tid = threadIdx.x;

    // Initialize shared variables
    if (tid == 0) {
        write_pos = 0;
        read_pos = 0;
        data_count = 0;
    }
    __syncthreads();

    // Producer threads (first half)
    if (tid < blockDim.x / 2) {
        for (int i = 0; i < 4; i++) {  // Each producer creates 4 items
            float data = tid * 4 + i;

            // Wait for buffer space
            while (data_count >= 128) {
                __syncthreads();
            }

            // Produce data
            int pos = atomicAdd(&write_pos, 1) % 128;
            buffer[pos] = data;
            atomicAdd(&data_count, 1);

            __syncthreads();
        }
    }
    // Consumer threads (second half)
    else {
        for (int i = 0; i < 2; i++) {  // Each consumer processes 2 items
            // Wait for data
            while (data_count <= 0) {
                __syncthreads();
            }

            // Consume data
            int pos = atomicAdd(&read_pos, 1) % 128;
            float data = buffer[pos];
            atomicSub(&data_count, 1);

            // Process data
            printf("Consumer %d processed: %.1f\n", tid, data);

            __syncthreads();
        }
    }
}
```

---

##  **Block-Level Synchronization**

Block-level synchronization enables cooperation among all threads in a thread block through shared memory and barrier operations.

###  **Thread Block Barriers**

#### **Basic `__syncthreads()` Usage:**
```cpp
// Comprehensive __syncthreads() examples
__global__ void syncthreads_examples() {
    __shared__ float shared_array[256];
    int tid = threadIdx.x;

    // Example 1: Data dependency synchronization
    shared_array[tid] = tid;
    __syncthreads();  // Ensure all threads have written their data

    // Now safe to read from any location
    float sum = shared_array[tid] + shared_array[(tid + 1) % blockDim.x];

    // Example 2: Multi-phase computation
    shared_array[tid] = sum;
    __syncthreads();  // Wait for all phase 1 computations

    // Phase 2: Use results from phase 1
    if (tid < blockDim.x / 2) {
        shared_array[tid] += shared_array[tid + blockDim.x / 2];
    }
    __syncthreads();  // Wait for phase 2

    // Example 3: Conditional synchronization (CAREFUL!)
    if (tid % 2 == 0) {
        shared_array[tid] *= 2.0f;
    }
    // All threads must reach this point, even if they didn't execute the if
    __syncthreads();
}
```

#### **Advanced Barrier Patterns:**
```cpp
// Custom barrier implementations for special cases
__device__ void partial_barrier(int thread_count) {
    __shared__ int barrier_counter;

    if (threadIdx.x == 0) {
        barrier_counter = 0;
    }
    __syncthreads();

    // Only specified number of threads participate
    if (threadIdx.x < thread_count) {
        atomicAdd(&barrier_counter, 1);

        // Busy wait until all participating threads arrive
        while (barrier_counter < thread_count) {
            // Spin wait
        }
    }

    __syncthreads();  // Ensure non-participating threads also wait
}

__global__ void custom_barrier_demo() {
    int tid = threadIdx.x;

    // Only first 64 threads participate in this computation
    if (tid < 64) {
        // Do some work...
        float result = sin(tid * 0.1f);
    }

    // Custom barrier for just these 64 threads
    partial_barrier(64);

    // Now all threads (including non-participating ones) can continue
    printf("Thread %d continuing after partial barrier\n", tid);
}
```

###  **Reduction Patterns**

#### **Block-Level Reduction:**
```cpp
// Efficient block-level reduction with multiple strategies
template<int BLOCK_SIZE>
__global__ void block_reduction_optimized(float* input, float* output, int N) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    shared_data[tid] = (global_id < N) ? input[global_id] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes result for this block
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

// Warp-optimized reduction (no sync needed within warp)
template<int BLOCK_SIZE>
__global__ void warp_optimized_reduction(float* input, float* output, int N) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Load and reduce in shared memory
    shared_data[tid] = (global_id < N) ? input[global_id] : 0.0f;
    __syncthreads();

    // Reduce to 32 elements (1 per warp)
    for (int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Final warp reduction (no __syncthreads needed)
    if (tid < 32) {
        // Manually unroll for efficiency
        if (BLOCK_SIZE >= 64) shared_data[tid] += shared_data[tid + 32];
        shared_data[tid] += shared_data[tid + 16];
        shared_data[tid] += shared_data[tid + 8];
        shared_data[tid] += shared_data[tid + 4];
        shared_data[tid] += shared_data[tid + 2];
        shared_data[tid] += shared_data[tid + 1];
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}
```

#### **Scan (Prefix Sum) Operations:**
```cpp
// Block-level exclusive scan implementation
__global__ void block_exclusive_scan(int* input, int* output, int N) {
    __shared__ int shared_data[256];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Load input data
    int value = (global_id < N) ? input[global_id] : 0;
    shared_data[tid] = value;
    __syncthreads();

    // Up-sweep phase (build sum tree)
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            shared_data[index] += shared_data[index - stride];
        }
        __syncthreads();
    }

    // Clear the last element (for exclusive scan)
    if (tid == 0) {
        shared_data[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase (distribute sums)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int temp = shared_data[index];
            shared_data[index] += shared_data[index - stride];
            shared_data[index - stride] = temp;
        }
        __syncthreads();
    }

    // Write results
    if (global_id < N) {
        output[global_id] = shared_data[tid];
    }
}
```

---

##  **Warp-Level Synchronization**

Warp-level synchronization leverages the SIMT execution model for high-performance intra-warp coordination.

###  **Implicit Warp Synchronization**

#### **SIMT Synchronization Properties:**
```cpp
// Demonstrate implicit warp synchronization
__global__ void implicit_warp_sync_demo() {
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // All threads in a warp execute in lockstep (SIMT)
    float value = lane_id * 2.0f;

    // This is safe within a warp - no explicit sync needed
    float neighbor = __shfl_sync(0xFFFFFFFF, value, (lane_id + 1) % 32);

    // Warp vote operations work without explicit sync
    bool condition = (value > 15.0f);
    int count = __popc(__ballot_sync(0xFFFFFFFF, condition));

    if (lane_id == 0) {
        printf("Warp %d: %d threads satisfy condition\n", warp_id, count);
    }

    // However, be careful with Volta+ independent thread scheduling
    // Always use __syncwarp() when thread divergence is possible
}
```

#### **Explicit Warp Synchronization:**
```cpp
// Modern warp synchronization with __syncwarp()
__global__ void explicit_warp_sync_demo() {
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // Volta+ architecture: threads can execute independently
    if (lane_id % 2 == 0) {
        // Even threads do extra work
        for (int i = 0; i < 100; i++) {
            float dummy = sin(i * 0.1f);
        }
    }

    // Explicit synchronization required for Volta+
    __syncwarp(0xFFFFFFFF);  // All threads in warp

    // Now all threads are guaranteed to be at the same point
    float value = lane_id;
    float sum = warp_reduce_sum(value);  // Safe to use warp primitives

    if (lane_id == 0) {
        printf("Warp sum: %.1f\n", sum);
    }
}

// Conditional warp synchronization
__global__ void conditional_warp_sync() {
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // Only sync active threads
    bool is_active = (lane_id < 16);
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, is_active);

    if (is_active) {
        // Work done by active threads
        float result = lane_id * 3.0f;

        // Sync only active threads
        __syncwarp(active_mask);

        // Warp operations on active subset
        float broadcast_value = __shfl_sync(active_mask, result, 0);

        printf("Lane %d received broadcast: %.1f\n", lane_id, broadcast_value);
    }
}
```

###  **Warp Cooperative Algorithms**

#### **Warp-Level Sorting:**
```cpp
// Bitonic sort within a warp
__device__ void warp_bitonic_sort(float* data, int warp_size = 32) {
    int lane_id = threadIdx.x % 32;
    float value = data[lane_id];

    // Bitonic sort network
    for (int stage = 2; stage <= warp_size; stage <<= 1) {
        for (int step = stage >> 1; step > 0; step >>= 1) {
            int partner = lane_id ^ step;
            float partner_value = __shfl_sync(0xFFFFFFFF, value, partner);

            bool ascending = ((lane_id & stage) == 0);
            bool should_swap = (ascending == (value > partner_value));

            if (should_swap) {
                value = partner_value;
            }
        }
    }

    data[lane_id] = value;
}

__global__ void warp_sort_demo(float* data, int N) {
    __shared__ float warp_data[32];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Each warp sorts its own 32 elements
    int global_idx = warp_id * 32 + lane_id;

    if (global_idx < N) {
        warp_data[lane_id] = data[global_idx];
    } else {
        warp_data[lane_id] = FLT_MAX;  // Padding
    }

    warp_bitonic_sort(warp_data);

    if (global_idx < N) {
        data[global_idx] = warp_data[lane_id];
    }
}
```

#### **Warp-Level Matrix Operations:**
```cpp
// Cooperative matrix multiplication within warp
__global__ void warp_matrix_multiply(float* A, float* B, float* C,
                                    int M, int N, int K) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Each warp computes one row of result matrix
    int row = warp_id;

    if (row < M) {
        // Each thread in warp handles multiple columns
        for (int col_base = 0; col_base < N; col_base += 32) {
            int col = col_base + lane_id;

            if (col < N) {
                float result = 0.0f;

                // Dot product computation
                for (int k = 0; k < K; k++) {
                    float a_val = A[row * K + k];  // Broadcast across warp
                    float b_val = B[k * N + col];  // Unique per thread
                    result += a_val * b_val;
                }

                C[row * N + col] = result;
            }
        }
    }
}
```

---

##  **Grid-Level Coordination**

Grid-level coordination requires kernel boundaries or specialized synchronization primitives for coordination across thread blocks.

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

---

##  **Atomic Operations**

Atomic operations provide thread-safe access to memory locations, enabling coordination without explicit barriers.

###  **Basic Atomic Operations**

#### **Atomic Arithmetic:**
```cpp
// Comprehensive atomic operations demonstration
__global__ void atomic_operations_demo(int* counters, float* values, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Integer atomic operations
        atomicAdd(&counters[0], 1);                    // Increment counter
        atomicMax(&counters[1], tid);                  // Track maximum
        atomicMin(&counters[2], tid);                  // Track minimum

        // Conditional atomic operations
        int old_val = atomicCAS(&counters[3], 0, tid); // Compare-and-swap
        if (old_val == 0) {
            printf("Thread %d was first to set counter[3]\n", tid);
        }

        // Floating-point atomics (newer architectures)
        atomicAdd(&values[0], tid * 0.1f);            // Float addition

        // Bitwise atomic operations
        atomicOr(&counters[4], 1 << (tid % 32));       // Set bit
        atomicAnd(&counters[5], ~(1 << (tid % 32)));   // Clear bit
        atomicXor(&counters[6], 1 << (tid % 32));      // Toggle bit
    }
}
```

#### **Custom Atomic Operations:**
```cpp
// Implement custom atomic operations using atomicCAS
__device__ float atomicAddFloat(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed,
                       __float_as_int(val + __int_as_float(assumed)));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// Atomic maximum for floating-point
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        float current = __int_as_float(assumed);
        float new_val = fmaxf(current, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);

    return __int_as_float(old);
}
```

###  **Atomic Performance Patterns**

#### **Reducing Atomic Contention:**
```cpp
// High contention - poor performance
__global__ void high_contention_atomics(int* global_counter, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // All threads compete for same memory location
        atomicAdd(global_counter, 1);  // High contention!
    }
}

// Reduced contention - better performance
__global__ void reduced_contention_atomics(int* global_counter, int N) {
    __shared__ int block_counter;
    int tid = threadIdx.x;

    // Initialize shared counter
    if (tid == 0) {
        block_counter = 0;
    }
    __syncthreads();

    // All threads in block increment shared counter
    if (threadIdx.x + blockIdx.x * blockDim.x < N) {
        atomicAdd(&block_counter, 1);
    }
    __syncthreads();

    // Only one thread per block updates global counter
    if (tid == 0) {
        atomicAdd(global_counter, block_counter);
    }
}

// Warp-level aggregation - best performance
__global__ void warp_aggregated_atomics(int* global_counter, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = tid % 32;

    if (tid < N) {
        // Each thread wants to add 1
        int contribution = 1;

        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            contribution += __shfl_down_sync(0xFFFFFFFF, contribution, offset);
        }

        // Only lane 0 performs atomic operation
        if (lane_id == 0) {
            atomicAdd(global_counter, contribution);
        }
    }
}
```

#### **Lock-Free Data Structures:**
```cpp
// Lock-free stack implementation
struct LockFreeStack {
    struct Node {
        int data;
        Node* next;
    };

    Node* head;

    __device__ void push(int value) {
        Node* new_node = new Node{value, nullptr};
        Node* old_head;

        do {
            old_head = head;
            new_node->next = old_head;
        } while (atomicCAS((unsigned long long*)&head,
                          (unsigned long long)old_head,
                          (unsigned long long)new_node) !=
                (unsigned long long)old_head);
    }

    __device__ bool pop(int& result) {
        Node* old_head;
        Node* new_head;

        do {
            old_head = head;
            if (old_head == nullptr) {
                return false;  // Stack is empty
            }
            new_head = old_head->next;
        } while (atomicCAS((unsigned long long*)&head,
                          (unsigned long long)old_head,
                          (unsigned long long)new_head) !=
                (unsigned long long)old_head);

        result = old_head->data;
        delete old_head;
        return true;
    }
};

__global__ void lock_free_stack_demo(LockFreeStack* stack, int* results, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Push phase
        stack->push(tid);

        __syncthreads();  // Ensure all pushes complete

        // Pop phase
        int value;
        if (stack->pop(value)) {
            results[tid] = value;
        } else {
            results[tid] = -1;  // Failed to pop
        }
    }
}
```

---

##  **Memory Consistency**

Understanding memory consistency models is crucial for correct synchronization across different memory types and scopes.

###  **Memory Ordering**

#### **Memory Fence Operations:**
```cpp
// Memory fence usage for consistency
__global__ void memory_consistency_demo(int* flags, int* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Producer thread
        if (tid % 2 == 0) {
            // Write data first
            data[tid] = tid * 100;

            // Memory fence ensures data write completes before flag write
            __threadfence();  // Global memory fence

            // Signal data is ready
            flags[tid] = 1;
        }
        // Consumer thread
        else {
            int partner = tid - 1;

            // Busy wait for flag
            while (atomicAdd(&flags[partner], 0) == 0) {
                // Spin wait
            }

            // Memory fence ensures flag read completes before data read
            __threadfence();

            // Now safe to read data
            int value = data[partner];
            printf("Thread %d read value %d\n", tid, value);
        }
    }
}

// Different fence scopes
__global__ void memory_fence_scopes(int* global_data, int N) {
    __shared__ int shared_data[256];
    int tid = threadIdx.x;

    // Thread fence (affects all memory spaces)
    shared_data[tid] = tid;
    global_data[tid] = tid * 2;
    __threadfence();  // Ensures both writes are visible

    // Block fence (affects memory visible to thread block)
    __threadfence_block();  // Ensures shared memory consistency

    // System fence (affects system-wide memory)
    __threadfence_system();  // Ensures visibility to CPU and other GPUs
}
```

#### **Relaxed Memory Model Programming:**
```cpp
// Handle relaxed memory ordering correctly
__global__ void relaxed_memory_programming(volatile int* sync_flags,
                                          int* shared_buffer, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_id = blockIdx.x;

    if (tid < N) {
        // Each block acts as producer or consumer
        if (block_id % 2 == 0) {
            // Producer block
            shared_buffer[tid] = tid + 1000;

            // Ensure write completes
            __threadfence();

            // Signal completion using volatile (prevents compiler optimization)
            sync_flags[block_id] = 1;
        }
        else {
            // Consumer block
            int producer_id = block_id - 1;

            // Wait for producer signal
            while (sync_flags[producer_id] == 0) {
                // Volatile access prevents optimization
            }

            // Read data produced by other block
            int consumed_value = shared_buffer[tid - blockDim.x];

            printf("Block %d thread %d consumed: %d\n",
                   block_id, threadIdx.x, consumed_value);
        }
    }
}
```

---

##  **Debugging Synchronization**

Synchronization bugs are among the most challenging to debug. Here are tools and techniques for identifying and fixing them.

###  **Common Synchronization Bugs**

#### **Race Condition Detection:**
```cpp
// Race condition example and detection
__global__ void race_condition_demo(int* shared_counter, int* race_detector, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // BUG: Race condition on shared_counter
        int old_value = *shared_counter;
        int new_value = old_value + 1;
        *shared_counter = new_value;

        // Record the values seen by each thread
        race_detector[tid * 2] = old_value;
        race_detector[tid * 2 + 1] = new_value;
    }
}

// Host-side race detection analysis
void analyze_race_conditions(int* h_race_detector, int N) {
    printf("Race Condition Analysis:\n");

    std::map<int, int> old_value_counts;
    std::map<int, int> new_value_counts;

    for (int i = 0; i < N; i++) {
        int old_val = h_race_detector[i * 2];
        int new_val = h_race_detector[i * 2 + 1];

        old_value_counts[old_val]++;
        new_value_counts[new_val]++;
    }

    printf("Old values seen: ");
    for (auto& p : old_value_counts) {
        printf("%d(%d times) ", p.first, p.second);
    }
    printf("\n");

    printf("New values written: ");
    for (auto& p : new_value_counts) {
        printf("%d(%d times) ", p.first, p.second);
    }
    printf("\n");

    // Check for evidence of race conditions
    if (old_value_counts.size() != N || new_value_counts.size() != N) {
        printf("RACE CONDITION DETECTED: Non-unique values observed!\n");
    }
}
```

#### **Deadlock Prevention:**
```cpp
// Potential deadlock scenario and prevention
__global__ void potential_deadlock_demo(int* resource1, int* resource2,
                                       int* lock1, int* lock2) {
    int tid = threadIdx.x;

    // BAD: Can cause deadlock
    if (tid % 2 == 0) {
        // Even threads acquire lock1 then lock2
        while (atomicCAS(lock1, 0, 1) != 0) { /* spin */ }
        while (atomicCAS(lock2, 0, 1) != 0) { /* spin */ }

        // Critical section
        *resource1 += 1;
        *resource2 += 1;

        atomicExch(lock2, 0);
        atomicExch(lock1, 0);
    } else {
        // Odd threads acquire lock2 then lock1 - DEADLOCK POTENTIAL!
        while (atomicCAS(lock2, 0, 1) != 0) { /* spin */ }
        while (atomicCAS(lock1, 0, 1) != 0) { /* spin */ }

        // Critical section
        *resource1 += 1;
        *resource2 += 1;

        atomicExch(lock1, 0);
        atomicExch(lock2, 0);
    }
}

// FIXED: Consistent lock ordering prevents deadlock
__global__ void deadlock_free_demo(int* resource1, int* resource2,
                                  int* lock1, int* lock2) {
    int tid = threadIdx.x;

    // Always acquire locks in same order: lock1 then lock2
    while (atomicCAS(lock1, 0, 1) != 0) { /* spin */ }
    while (atomicCAS(lock2, 0, 1) != 0) { /* spin */ }

    // Critical section
    *resource1 += tid;
    *resource2 += tid * 2;

    // Release in reverse order
    atomicExch(lock2, 0);
    atomicExch(lock1, 0);
}
```

###  **Synchronization Testing Framework**

#### **Stress Testing Synchronization:**
```cpp
// Framework for stress testing synchronization primitives
class SyncStressTester {
private:
    int* d_test_data;
    int* d_results;
    int* d_error_count;
    int N;

public:
    SyncStressTester(int size) : N(size) {
        cudaMalloc(&d_test_data, N * sizeof(int));
        cudaMalloc(&d_results, N * sizeof(int));
        cudaMalloc(&d_error_count, sizeof(int));
    }

    void test_atomic_correctness(int iterations) {
        for (int iter = 0; iter < iterations; iter++) {
            // Reset data
            cudaMemset(d_test_data, 0, N * sizeof(int));
            cudaMemset(d_error_count, 0, sizeof(int));

            // Launch stress test
            dim3 blocks((N + 255) / 256);
            dim3 threads(256);

            atomic_stress_test<<<blocks, threads>>>(d_test_data, d_results,
                                                   d_error_count, N);
            cudaDeviceSynchronize();

            // Check results
            int h_error_count;
            cudaMemcpy(&h_error_count, d_error_count, sizeof(int),
                      cudaMemcpyDeviceToHost);

            if (h_error_count > 0) {
                printf("Iteration %d: %d errors detected!\n", iter, h_error_count);
            }
        }
    }

    ~SyncStressTester() {
        cudaFree(d_test_data);
        cudaFree(d_results);
        cudaFree(d_error_count);
    }
};

__global__ void atomic_stress_test(int* data, int* results, int* error_count, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Each thread increments its assigned location many times
        for (int i = 0; i < 1000; i++) {
            atomicAdd(&data[tid % 100], 1);  // Contended updates
        }

        __syncthreads();

        // Verify results
        if (tid < 100) {
            int expected = (N / 100) * 1000;  // Expected value
            if (data[tid] != expected) {
                atomicAdd(error_count, 1);
                printf("Error at index %d: expected %d, got %d\n",
                       tid, expected, data[tid]);
            }
        }
    }
}
```

---

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

- **Next Step**: [ Execution Constraints Guide](1e_execution_constraints_guide.md) - Resource limits and workarounds
- **Foundation**: [ Streaming Multiprocessors Guide](1c_streaming_multiprocessors_deep.md) - SM architecture and scheduling
- **Parallelism**: [ Warp Execution Guide](1b_warp_execution.md) - Warp-level coordination
- **Overview**: [ Execution Model Overview](1_cuda_execution_model.md) - Quick reference and navigation

---

** Pro Tip**: Synchronization is about coordination, not just correctness. Choose the right synchronization primitive for both correctness AND performance!
