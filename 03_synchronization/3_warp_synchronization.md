#  Warp-Level Synchronization

Warp-level synchronization leverages the SIMT execution model for high-performance intra-warp coordination.

**Previous: [Block-Level Synchronization](2_block_synchronization.md)** | **Next: [Grid-Level Coordination](4_grid_coordination.md)**

---

##  **Warp-Level Synchronization**

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
