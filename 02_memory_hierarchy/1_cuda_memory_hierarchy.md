#  CUDA - **[Global Memory Optimization](2_global_memory.md)** - Coalescing patterns and performance analysis
- **[Shared Memory Complete Guide](3_shared_memory.md)** - Matrix multiplication, bank conflicts, benchmarks
- **[Constant Memory Complete Guide](4_constant_memory.md)** - Domain examples, optimization strategies
- **[Unified Memory Complete Guide](5_unified_memory.md)** - Advanced techniques, multi-GPU, performance analysis
- **[Memory Debugging Toolkit](6_memory_debugging.md)** - Troubleshooting workflows and profiling strategiesy Hierarchy - Overview & Quick Reference

Understanding the CUDA memory hierarchy is **critical** for writing performant kernels. The choice of memory and access pattern dramatically impacts latency, bandwidth utilization, and overall throughput.

##  Navigation Guide

###  **Detailed Section Files**
- **[Global Memory Optimization](2_global_memory.md)** - Coalescing patterns and performance analysis
- **[Shared Memory Complete Guide](3_shared_memory.md)** - Matrix multiplication, bank conflicts, benchmarks
- **[Constant Memory Complete Guide](4_constant_memory.md)** - Domain examples, optimization strategies
- **[Unified Memory Complete Guide](5_unified_memory.md)** - Advanced techniques, multi-GPU, performance analysis
- **[Memory Debugging Toolkit](6_memory_debugging.md)** - Troubleshooting workflows and profiling strategies

---

##  **Memory Types Complete Reference**

| Memory Type | Latency | Size Limit | Scope | Lifetime | Access Pattern | Best For | Detailed Guide |
|-------------|---------|------------|-------|----------|----------------|----------|----------------|
| **Registers** | ~1 cycle | 32-255/thread | Thread | Kernel | Private to thread | Scalars, loop counters, temporary values | This guide |
| **Local Memory** | 300-600 cycles | Unlimited | Thread | Kernel | Private (in global mem) | Arrays/structs too large for registers | This guide |
| **Shared Memory** | ~2-3 cycles | 48-164 KB/block | Block | Kernel | Shared within block | Tile storage, cooperative block computation | [Shared Guide](3_shared_memory.md) |
| **Global Memory** | 300-600 cycles | GPU VRAM | All Threads | Application | Read/Write by all | Main memory for large input/output datasets | [Global Guide2_global_memory.md) |
| **Constant Memory** | ~1 cycle (cached) | 64 KB total | All Threads | Application | Broadcast to all (RO) | Lookup tables, config params, small readonly data | [Constant Guide](4_constant_memory.md) |
| **Texture/Surface** | ~1-10 cycles (cached) | GPU VRAM | All Threads | Application | Specialized read/write | 2D/3D spatial data, image processing | This guide |
| **Unified Memory** | Variable | System RAM+GPU | CPU + GPU | Application | Auto-migrated host/device | Rapid development, shared CPU-GPU access | [Unified Guide](5_unified_memory.md) |


####  Registers
Registers provide the most efficient, low-latency storage available to CUDA threads. They are allocated per thread and are ideal for **frequently reused** values such as loop counters, temporary calculations, and scalar variables. However, registers are limited in number per thread, and excess usage causes register spills into **local memory**, which significantly degrades performance.

####  Local Memory
Despite its name, local memory actually resides in **global memory** and has similar high latency characteristics. This memory type is used for large thread-local arrays and register spills when the number of registers exceeds the hardware limit. Accesses to local memory are slower and uncached unless explicitly optimized, making it important to minimize its usage.

####  Shared Memory
Shared memory resides **on-chip** and provides very low latency access. It is visible and writable to all threads within a block, making it perfect for tiling algorithms (such as matrix multiplication) and inter-thread communication within a block. To achieve optimal performance, developers must carefully manage **bank conflicts** that can serialize memory accesses.

####  Global Memory
Global memory serves as the primary GPU memory (DRAM) and has high latency of approximately 300–600 cycles. It is accessible by all threads and can be accessed by the CPU through explicit memory copies or unified memory. The key optimization for global memory is ensuring **coalesced access** patterns to reduce transaction overhead and maximize memory bandwidth utilization.

####  Constant Memory
Constant memory is read-only memory that is cached and broadcast-efficient across all threads. It is size-limited (typically 64 KB) and is ideal for storing uniform values such as configuration parameters, coefficients, and lookup tables. This memory type is accessed via the `__constant__` qualifier and provides excellent performance when the same data is read by multiple threads.

####  Texture/Surface Memory
Texture and surface memory are specialized for 2D/3D spatial access patterns and are commonly used in image and signal processing applications. They offer built-in **caching**, **addressing**, and **interpolation** features that can significantly improve performance for spatially coherent data access patterns.

####  Unified Memory
Unified memory abstracts away explicit `cudaMemcpy` operations by providing system-managed page migration between CPU and GPU. While this makes programming easier, performance can suffer from page faults and unpredictable behavior if memory is not properly prefetched using techniques like `cudaMemPrefetchAsync()`.


###  Why It Matters

The choice of memory directly influences performance and developer productivity.

| Goal                           | Memory Type Recommendation                          |
|--------------------------------|-----------------------------------------------------|
| Maximum speed per thread       | Registers                                            |
| Intra-block communication      | Shared Memory                                        |
| Global data sharing            | Global or Unified Memory                             |
| Read-only broadcast values     | Constant Memory                                      |
| Host-device unified access     | Unified Memory                                       |
| 2D/3D data with locality       | Texture / Surface Memory                             |

Understanding these trade-offs enables you to:
- Avoid unnecessary latency.
- Exploit fast memory tiers (registers/shared).
- Design scalable, high-performance kernels.


##  **Performance Quick Reference**

###  **Memory Bandwidth Efficiency**
```
Theoretical Peak: ~1000 GB/s (RTX 4090)

Achieved Performance by Optimization:
 Naive Global Access:     124 GB/s (12% efficiency)
 Coalesced Access:        201 GB/s (20% efficiency)
 Shared Memory Tiled:     487 GB/s (48% efficiency)
 Optimized + No Conflicts: 603 GB/s (60% efficiency)
```

###  **Common Optimization Impact**
| Optimization | Typical Speedup | When to Use |
|--------------|----------------|-------------|
| **Coalesced Access** | 1.5-3x | Always for global memory |
| **Shared Memory Tiling** | 3-8x | Reused data within blocks |
| **Bank Conflict Fix** | 1.2-2x | Column-wise shared access |
| **Constant Memory** | 2-50x | Small uniform read-only data |
| **Unified Memory + Prefetch** | 1.5-4x | CPU-GPU workflows |

##  **Essential Code Patterns**

###  **Coalesced Global Memory Access**
```cpp
// Good: Consecutive threads → consecutive addresses
int idx = threadIdx.x + blockIdx.x * blockDim.x;
float val = data[idx];  // Perfect coalescing
```

###  **Shared Memory Template**
```cpp
__global__ void optimized_kernel(float* input, float* output) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 prevents bank conflicts

    // Load phase
    tile[threadIdx.y][threadIdx.x] = input[...]; // Coalesced load
    __syncthreads();

    // Compute phase
    float result = 0;
    for (int i = 0; i < TILE_SIZE; i++) {
        result += tile[threadIdx.y][i] * tile[i][threadIdx.x]; // Fast shared access
    }
    __syncthreads();

    // Store phase
    output[...] = result; // Coalesced store
}
```

###  **Constant Memory Pattern**
```cpp
__constant__ float lookup_table[1024];

// Host: Initialize once
cudaMemcpyToSymbol(lookup_table, host_data, sizeof(host_data));

// Device: Fast broadcast access
__device__ float fast_lookup(int index) {
    return lookup_table[index]; // All threads get same value efficiently
}
```

###  **Unified Memory Pattern**
```cpp
// Allocation
float* data;
cudaMallocManaged(&data, size);

// Optimization hints
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, 0); // GPU
cudaMemPrefetchAsync(data, size, 0); // Move to GPU before kernel

// Use from both CPU and GPU without explicit copies
kernel<<<blocks, threads>>>(data);
process_results_cpu(data);
```

##  **Common Performance Issues & Quick Fixes**

###  **Issue: Poor Global Memory Performance**
**Symptom:** `gld_efficiency < 50%` in profiler
**Quick Fix:** Ensure consecutive threads access consecutive memory
```cpp
// Bad: Strided access
data[threadIdx.x * stride]

// Good: Sequential access
data[threadIdx.x + blockIdx.x * blockDim.x]
```

###  **Issue: Shared Memory Bank Conflicts**
**Symptom:** `shared_load_transactions_per_request > 1.0`
**Quick Fix:** Add padding to shared arrays
```cpp
// Bad: 32-way conflicts on column access
__shared__ float tile[32][32];

// Good: Conflict-free with padding
__shared__ float tile[32][33]; // +1 element padding
```

###  **Issue: Unified Memory Page Faults**
**Symptom:** Irregular first-run performance
**Quick Fix:** Prefetch before kernel launch
```cpp
cudaMemPrefetchAsync(data, size, 0); // Move to GPU
kernel<<<blocks, threads>>>(data);
```

##  **Profiling Quick Commands**

###  **Memory Coalescing Check**
```bash
ncu --metrics gld_efficiency,gst_efficiency ./app
# Target: > 80%
```

###  **Shared Memory Conflicts**
```bash
ncu --metrics shared_load_transactions_per_request ./app
# Target: < 1.1
```

###  **Overall Memory Performance**
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./app
# Target: > 60% for memory-bound kernels
```

###  **Unified Memory Analysis**
```bash
nsys profile --trace=cuda ./app
# Look for: Page fault patterns in timeline view
```

##  **Optimization Decision Tree**

```
Memory Performance Issue?
 Low global memory efficiency?
    Fix coalescing → [Global Guide2_global_memory.md)
    Use shared memory tiling → [Shared Guide](3_shared_memory.md)
 Shared memory bank conflicts?
    Add padding, fix access patterns → [Shared Guide&shared_memory.md#bank-conflicts)
 Small uniform read-only data?
    Use constant memory → [Constant Guide](4_constant_memory.md)
 Complex CPU-GPU workflows?
    Optimize unified memory → [Unified Guide](5_unified_memory.md)
 Still having issues?
     Systematic debugging → [Debug Guide6_memory_debugging.md)
```

##  **Key Principles**

1. ** Coalesce First**: Always ensure global memory accesses are coalesced
2. ** Reuse in Shared**: Move frequently reused data to shared memory
3. ** Broadcast Constants**: Use constant memory for uniform read-only data
4. ** Profile Everything**: Use Nsight Compute/Systems to validate optimizations
5. ** Understand Trade-offs**: Sometimes lower occupancy = better performance

##  **Next Steps**

1. **Start with**: [Global Memory Guide2_global_memory.md) for coalescing basics
2. **Then learn**: [Shared Memory Guide](3_shared_memory.md) for advanced tiling
3. **Specialize with**: [Constant4_constant_memory.md) or [Unified5_unified_memory.md) memory guides
4. **Debug with**: [Debugging Toolkit](6_memory_debugging.md) when performance issues arise

---

** Pro Tip**: Start simple with coalesced global memory, then progressively add shared memory optimizations. Profile at each step to quantify improvements!


