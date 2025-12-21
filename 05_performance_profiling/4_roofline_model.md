#  Roofline Model

**Previous: [Nsight Compute](3_nsight_compute.md)** | **Next: [Typical Optimization Strategies](5_optimization_strategies.md)**

---

##  **Roofline Model**

The **Roofline Model** is a visual and conceptual tool that helps you understand the performance limits of your CUDA kernel based on **compute throughput vs. memory bandwidth**.

It shows whether your kernel is:
- **Memory-bound**: Limited by how fast data can be moved from memory (DRAM or cache) to compute units.
- **Compute-bound**: Limited by how fast arithmetic instructions can be executed by the SMs.


###  Key Concepts

| Concept               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **FLOP/s**             | Floating-point operations per second – a measure of computational intensity |
| **Operational Intensity (OI)** | Ratio of FLOPs to bytes accessed (FLOPs / Byte)                        |
| **Bandwidth Ceiling**  | The maximum memory bandwidth of the GPU (e.g., 1.6 TB/s on H100)             |
| **Compute Roofline**   | The theoretical peak compute throughput (e.g., 60 TFLOP/s FP32)              |

The intersection of your kernel’s **OI** and the roofline curve tells you what your performance ceiling is — and whether you’re bandwidth- or compute-limited.

###  Visualization

          Compute-bound region
              /
             /
            /   <-- Compute Roofline (FLOPs/sec)
           /
          /
         /
        /___________________
       /|
      / |                   Memory-bound region
     /  |
    /   |
   /    |   <-- Your Kernel
  /



- Kernels **below the roofline** are under-optimized.
- Move **up/right** by increasing data reuse or parallelism.

###  Optimization Paths

####  Memory-Bound Kernels

These are limited by bandwidth — they spend more time waiting on memory than doing useful computation.

##### Strategies:
- **Improve memory coalescing**: Align memory access patterns across warps.
- **Use shared memory**: Temporarily cache data to reduce global memory loads.
- **Minimize transfers**: Avoid redundant memory reads/writes.
- **Prefetch memory**: Use `cudaMemPrefetchAsync()` for Unified Memory.

####  Compute-Bound Kernels

These are limited by ALU throughput — they max out the FP units before exhausting memory.

##### Strategies:
- **Increase ILP**: Schedule multiple independent instructions to reduce stalls.
- **Reduce warp divergence**: Ensure uniform control flow across threads.
- **Loop unrolling**: Reduces instruction dispatch overhead and enables more aggressive compiler optimizations.
- **Use fast math**: `__fmul_rn`, `__fmaf_rn`, or even FP16/Tensor Cores when precision allows.


###  How to Use in Practice

1. Use `Nsight Compute` to calculate:
   - FLOPs executed
   - Bytes transferred from global memory
   - → Compute **Operational Intensity (OI)** = FLOPs / Bytes
2. Plot OI vs. achieved FLOPs/s and compare to GPU roofline.
3. Decide whether to optimize for memory or compute.


###  Why It Matters

The Roofline Model helps you:
- Avoid over-optimizing in the wrong dimension (e.g., tuning FP math when memory is the bottleneck)
- Target realistic performance ceilings
- Prioritize changes that **move your kernel upward** on the roofline chart


The Roofline Model gives CUDA developers **a clear, visual way to reason about performance limits**, especially when navigating complex trade-offs between memory and compute.
