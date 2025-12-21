#  Detecting Bottlenecks

**Previous: [Kernel Launch Config Tuning](7_kernel_launch_tuning.md)**

---

##  **Detecting Bottlenecks**

Efficient CUDA performance requires identifying and addressing the **right bottleneck**. Profiling tools like **Nsight Compute** and **Nsight Systems** help correlate low-level metrics with performance issues.

---

###  Common Bottlenecks and Their Root Causes

| Symptom                 | Likely Cause                                                        |
|-------------------------|---------------------------------------------------------------------|
| **Low SM Occupancy**    | - Excessive **register** or **shared memory** usage per thread/block <br> - Too few threads per block or grid-size mismatch |
| **High Memory Latency** | - **Poor memory coalescing** <br> - **Unaligned memory accesses** <br> - Relying too much on global memory instead of shared memory |
| **Warp Execution < 100%** | - **Branch divergence**: Threads in a warp follow different paths <br> - Uneven workload distribution |
| **L2 Cache Miss Rate High** | - Large working set exceeding cache capacity <br> - No **temporal or spatial locality** <br> - Ineffective data reuse patterns |

---

###  How to Detect

- Use **Nsight Compute** to monitor:
  - SM utilization
  - Warp execution efficiency
  - Global load efficiency
  - Cache hit rates

- Use **`--metrics`** with `nvprof` or `ncu`:
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./my_kernel
```

###  Actionable Tips
- Low Occupancy → Reduce register/shared mem usage or increase threads/block
- Memory Bottlenecks → Improve coalescing, leverage shared memory
- Warp Inefficiency → Restructure control flow to minimize divergence
- Cache Misses → Refactor for reuse or reduce working set size
