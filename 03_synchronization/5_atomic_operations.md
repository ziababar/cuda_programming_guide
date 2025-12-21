#  Atomic Operations

Atomic operations provide thread-safe access to memory locations, enabling coordination without explicit barriers.

**Previous: [Grid-Level Coordination](4_grid_coordination.md)** | **Next: [Memory Consistency](6_memory_consistency.md)**

---

##  **Atomic Operations**

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
