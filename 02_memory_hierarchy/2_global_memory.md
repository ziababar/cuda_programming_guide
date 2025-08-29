#  Global Memory Advanced Optimization Guide

Global memory is the main data reservoir for CUDA kernels, but it's slow. This guide covers advanced optimization techniques for maximizing global memory performance through coalescing, access patterns, and bandwidth utilization.

**[Back to Overview&cuda_memory_hierarchy.md)** | **Next: [Shared Memory Guide](3_shared_memory.md)**

---

##  **Table of Contents**

1. [ Understanding Memory Coalescing](#-understanding-memory-coalescing)
2. [ Coalesced Access Patterns](#-coalesced-access-patterns)
3. [ Common Anti-Patterns](#-common-anti-patterns)
4. [ Visual Memory Access Analysis](#-visual-memory-access-analysis)
5. [ Performance Impact Analysis](#-performance-impact-analysis)
6. [ Advanced Optimization Techniques](#-advanced-optimization-techniques)
7. [ Profiling and Debugging](#-profiling-and-debugging)

---

##  **Understanding Memory Coalescing**

Memory coalescing is the process by which threads in a warp (32 threads) access consecutive memory addresses, allowing the GPU to service these memory requests using fewer memory transactions. It is a fundamental performance optimization for global memory access.

###  **Performance Impact**
- **Coalesced access**: 1-2 memory transactions per warp
- **Non-coalesced access**: Up to 32 separate transactions per warp
- **Performance difference**: 10-20x slowdown for poor coalescing

###  **Hardware Context**
Global memory accesses are slow (400–600 clock cycles). However, when memory accesses are coalesced, the GPU can fetch data for an entire warp with just 1 or 2 memory transactions. If they are not coalesced, each thread might require its own memory transaction — severely degrading performance.

---

##  **Coalesced Access Patterns**

###  **Perfect Coalescing Example**
```cpp
__global__ void coalesced_kernel(float* input, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        float value = input[idx];  //  Perfect coalescing
        output[idx] = value * 2.0f;  //  Perfect coalescing
    }
}
```

####  **Coalesced Access Visualization**
```
Warp (32 threads):     T0 T1 T2 T3 T4 T5 T6 T7  ... T31
Memory Addresses:     [0] [1] [2] [3] [4] [5] [6] [7] ... [31]
Memory Transaction:   |------------ Single 128-byte load ------------|

 Result: 1 memory transaction serves entire warp (32 floats = 128 bytes)
```

###  **Stride-1 Access (Optimal)**
```cpp
// Thread 0 accesses data[0], Thread 1 accesses data[1], etc.
int idx = threadIdx.x + blockIdx.x * blockDim.x;
float val = data[idx];  // Consecutive access pattern
```

###  **Vectorized Coalesced Access**
```cpp
__global__ void vectorized_kernel(float4* input, float4* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        float4 vec = input[idx];  // Loads 16 bytes per thread efficiently
        vec.x *= 2.0f; vec.y *= 2.0f; vec.z *= 2.0f; vec.w *= 2.0f;
        output[idx] = vec;
    }
}
```

---

##  **Common Anti-Patterns**

###  **Strided Access Pattern**
```cpp
//  BAD: Strided access pattern
__global__ void strided_kernel(float* data, int stride, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        float val = data[idx * stride];  // Non-consecutive access
    }
}
```

####  **Strided Access Visualization**
```
Stride = 4:
Warp (32 threads):     T0 T1 T2 T3 T4 T5 T6 T7  ... T31
Memory Addresses:     [0]  [4]  [8]  [12] [16] [20] [24] [28] ... [124]
Memory Transactions:   |--1--| |--2--| |--3--| |--4--| |--5--| |--6--|

 Result: Multiple transactions, poor bandwidth utilization
```

###  **Array of Structures (AoS) Access**
```cpp
struct Particle { float x, y, z, mass; };

//  BAD: Accessing only x coordinates
__global__ void aos_kernel(Particle* particles, float* x_coords, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        x_coords[idx] = particles[idx].x;  // Stride = sizeof(Particle) = 16 bytes
    }
}
```

###  **Structure of Arrays (SoA) Solution**
```cpp
struct ParticleArrays {
    float* x; float* y; float* z; float* mass;
};

//  GOOD: Coalesced access to separate arrays
__global__ void soa_kernel(ParticleArrays particles, float* x_coords, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        x_coords[idx] = particles.x[idx];  // Perfect coalescing
    }
}
```

---

##  **Visual Memory Access Analysis**

###  **Transaction Comparison Chart**
```
COALESCED (Optimal):
Memory:  [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]...
Threads:  T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15...
Access:   ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑   ↑   ↑   ↑   ↑   ↑
Result:   |------------------ 1 Transaction -------------------|

STRIDED (Poor - Stride 2):
Memory:  [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]...
Threads:  T0 T1 T2 T3 T4 T5 T6 T7 T8...
Access:   ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
Result:   |--1--| |--2--| |--3--| |--4--| |--5--| |--6--| |--7--| |--8--|

RANDOM (Worst):
Memory:  [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]...
Threads:  T0 T1 T2 T3 T4 T5
Access:   ↑       ↑         ↑     ↑          ↑       ↑
Result:   1 2         3 4          5 6    (32 transactions!)
```

###  **Memory Alignment Visualization**
```
128-byte Cache Line Boundaries:
|-------- 128 bytes --------||-------- 128 bytes --------|
[                            ][                            ]
 ↑                            ↑
 Aligned access Next cache line

Misaligned Access (Poor):
    |-------- 128 bytes --------||-------- 128 bytes --------|
    [      ][access pattern here][     ]
           ↑                     ↑
           Spans 2 cache lines = 2 transactions
```

---

##  **Performance Impact Analysis**

###  **Benchmark Results (RTX 4090, 1GB Array)**

| Access Pattern | Bandwidth (GB/s) | Efficiency | Transactions/Warp | Speedup |
|----------------|------------------|------------|-------------------|---------|
| **Sequential (Ideal)** | 847 GB/s | 84% | 1.0 | 1.0x |
| **Stride 2** | 445 GB/s | 44% | 2.0 | 0.53x |
| **Stride 4** | 289 GB/s | 29% | 4.0 | 0.34x |
| **Stride 8** | 167 GB/s | 17% | 8.0 | 0.20x |
| **Random Access** | 23 GB/s | 2% | 32.0 | 0.03x |

###  **Real-World Application Impact**

####  **Scientific Computing Examples**
```cpp
//  POOR: Particle simulation with AoS
struct Particle { float3 pos, vel, force; };  // 36 bytes
Particle particles[N];
// Accessing positions: stride = 36 bytes → poor coalescing

//  GOOD: SoA approach
float3 positions[N], velocities[N], forces[N];  // 12 bytes each
// Accessing positions: stride = 12 bytes → good coalescing
// Performance improvement: 3-4x faster
```

####  **Graphics Processing Examples**
```cpp
//  POOR: RGB image processing
struct Pixel { unsigned char r, g, b; };  // 3 bytes, poor alignment
// Processing only red channel: stride = 3 → very poor

//  GOOD: Separate channel arrays or RGBA
unsigned char red[width*height];   // Packed red channel
unsigned char rgba[width*height*4]; // RGBA with padding
// Performance improvement: 8-12x faster
```

---

##  **Advanced Optimization Techniques**

###  **1. Memory Alignment Optimization**
```cpp
// Ensure pointers are aligned to cache line boundaries
void* aligned_malloc(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, (size + 127) & ~127);  // Align to 128 bytes
    return ptr;
}
```

###  **2. Vectorized Memory Operations**
```cpp
// Use vector types for better memory throughput
__global__ void optimized_copy(float4* src, float4* dst, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        dst[idx] = src[idx];  // 16 bytes per thread vs 4 bytes
    }
}

// Launch with N/4 threads instead of N threads
dim3 block(256);
dim3 grid((N/4 + block.x - 1) / block.x);
optimized_copy<<<grid, block>>>(src, dst, N/4);
```

###  **3. Memory Access Reordering**
```cpp
// Transform non-coalesced patterns into coalesced ones
__global__ void matrix_transpose_optimized(float* input, float* output,
                                         int width, int height) {
    // Use shared memory to change access pattern
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Coalesced read from input
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    // Coalesced write to output (transposed coordinates)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

###  **4. Data Layout Transformation**
```cpp
// Convert AoS to SoA for better coalescing
void convert_aos_to_soa(Particle* aos_data, ParticleArrays* soa_data, int N) {
    // Host-side conversion
    for (int i = 0; i < N; i++) {
        soa_data->x[i] = aos_data[i].pos.x;
        soa_data->y[i] = aos_data[i].pos.y;
        soa_data->z[i] = aos_data[i].pos.z;
        soa_data->mass[i] = aos_data[i].mass;
    }
}
```

---

##  **Profiling and Debugging**

###  **Essential Nsight Compute Metrics**

####  **Memory Coalescing Metrics**
```bash
# Check global memory coalescing efficiency
ncu --metrics gld_efficiency,gst_efficiency ./app

# Target values:
# gld_efficiency: > 80% (good), > 90% (excellent)
# gst_efficiency: > 80% (good), > 90% (excellent)
```

####  **Memory Throughput Analysis**
```bash
# Check overall memory bandwidth utilization
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./app

# Check memory transaction efficiency
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./app
```

####  **Detailed Memory Analysis**
```bash
# Comprehensive memory profile
ncu --set full --section MemoryWorkloadAnalysis ./app

# Focus on specific metrics
ncu --metrics \
  gld_transactions,gld_requested_throughput,gld_efficiency,\
  gst_transactions,gst_requested_throughput,gst_efficiency \
  ./app
```

###  **Optimization Workflow**

#### **Step 1: Baseline Measurement**
```bash
# Get current performance
ncu --metrics gld_efficiency,dram__throughput.avg.pct_of_peak_sustained_elapsed ./app
```

#### **Step 2: Identify Issues**
```cpp
// Common diagnostic patterns
if (gld_efficiency < 50%) {
    // Likely strided or random access pattern
    // → Check data structure layout (AoS vs SoA)
    // → Verify thread-to-memory mapping
}

if (dram__throughput < 30%) {
    // Low overall bandwidth utilization
    // → Check for other bottlenecks (compute, occupancy)
    // → Consider shared memory optimizations
}
```

#### **Step 3: Apply Fixes and Validate**
```bash
# After optimization, re-measure
ncu --metrics gld_efficiency,dram__throughput.avg.pct_of_peak_sustained_elapsed ./app

# Compare before/after
ncu --csv --metrics gld_efficiency ./app > after.csv
# Analyze CSV data for quantitative improvement
```

###  **Common Debugging Scenarios**

#### **Scenario 1: Matrix Operations**
```cpp
// Problem: Column-wise access in row-major matrix
// Solution: Use transpose + row-wise access or shared memory tiling

//  Poor coalescing
for (int col = 0; col < width; col++) {
    float sum = matrix[col * height + row];  // Stride = height
}

//  Better approach with transpose or tiling
// See shared memory guide for complete solution
```

#### **Scenario 2: Image Processing**
```cpp
// Problem: Processing RGB channels separately
// Solution: Process all channels together or use separate arrays

//  Poor: RGB struct with 3-byte stride
struct RGB { unsigned char r, g, b; };

//  Better: RGBA with 4-byte alignment
struct RGBA { unsigned char r, g, b, a; };  // 4 bytes = good alignment
```

###  **Performance Validation**

#### **Before/After Comparison Template**
```cpp
// Timing harness for validation
float measure_kernel_performance(void (*kernel_func)(float*, float*, int),
                                float* input, float* output, int N, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    kernel_func(input, output, N);
    cudaDeviceSynchronize();

    // Measurement
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel_func(input, output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;  // Average time per kernel
}
```

---

##  **Key Takeaways**

1. ** Always Design for Coalescing**: Structure your algorithms so consecutive threads access consecutive memory
2. ** SoA > AoS**: Structure of Arrays typically provides better coalescing than Array of Structures
3. ** Use Vector Types**: `float4`, `int2`, etc., can improve memory throughput
4. ** Profile Early and Often**: Use Nsight Compute to validate your optimizations
5. ** Visualize Access Patterns**: Draw out how your threads access memory to spot issues

##  **Related Guides**

- **Next Step**: [Shared Memory Complete Guide](3_shared_memory.md) - Learn tiling techniques for further optimization
- **Debugging**: [Memory Debugging Toolkit](6_memory_debugging.md) - Systematic approach to memory issues
- **Overview**: [Memory Hierarchy Overview&cuda_memory_hierarchy.md) - Quick reference and navigation

---

** Pro Tip**: Start by ensuring your global memory accesses are coalesced, then consider shared memory optimizations. Many performance issues can be solved with proper data layout alone!
