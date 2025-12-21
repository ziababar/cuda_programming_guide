#  Block-Level Synchronization

Block-level synchronization enables cooperation among all threads in a thread block through shared memory and barrier operations.

**Previous: [Synchronization Fundamentals](1_synchronization_basics.md)** | **Next: [Warp-Level Synchronization](3_warp_synchronization.md)**

---

##  **Block-Level Synchronization**

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
