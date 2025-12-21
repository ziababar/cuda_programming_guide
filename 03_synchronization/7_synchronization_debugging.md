#  Debugging Synchronization

Synchronization bugs are among the most challenging to debug. Here are tools and techniques for identifying and fixing them.

**Previous: [Memory Consistency](6_memory_consistency.md)** | **Next: [Advanced Patterns](8_advanced_synchronization.md)**

---

##  **Debugging Synchronization**

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
