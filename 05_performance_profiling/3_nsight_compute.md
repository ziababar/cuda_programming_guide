#  Nsight Compute – Kernel Performance

**Previous: [Nsight Systems](2_nsight_systems.md)** | **Next: [Roofline Model](4_roofline_model.md)**

---

##  **Nsight Compute – Kernel Performance**

`Nsight Compute` is NVIDIA's low-level, per-kernel profiler. It provides **fine-grained performance metrics** that reveal how your CUDA kernel behaves at the microarchitectural level.

You can analyze memory throughput, warp execution efficiency, cache utilization, instruction mix, and more — all specific to a single kernel launch.


###  Why Use Nsight Compute?

- Drill down into individual kernels that appear slow in Nsight Systems.
- Identify causes of low throughput: memory issues, control flow divergence, register pressure.
- Validate optimization strategies by comparing metrics pre/post changes.


###  Key Metrics and What They Reveal

| Metric                        | Insight                                                                 |
|-------------------------------|-------------------------------------------------------------------------|
| **SM Efficiency**             | The percentage of time at least one warp was active on the SM. <br> **High values (80–100%)** indicate good GPU utilization.<br> **Low values** suggest idle SMs due to under-filled warps or excessive waiting. |
| **Occupancy**                 | Ratio of active warps to the max possible on an SM.<br> Helps hide memory latency.<br> Low occupancy may point to too many registers or shared memory usage. |
| **Warp Execution Efficiency** | Percentage of threads in a warp that are active.<br> Close to 100% = uniform execution.<br> Lower values indicate warp divergence (e.g., `if/else`, loop branching). |
| **Global Load Efficiency**    | Measures how well memory loads are coalesced and aligned.<br> High efficiency means consecutive threads access consecutive memory locations.<br> Poor efficiency leads to increased memory transactions and latency. |
| **L2 Cache Hit Rate**         | Indicates whether memory accesses benefit from caching.<br> High hit rate = good data reuse.<br> Low hit rate = working set is too large or poorly localized. |


###  Additional Useful Metrics

| Metric                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **DRAM Throughput**      | Total bandwidth usage – helps determine if the kernel is memory-bound.     |
| **Achieved Occupancy**   | Actual number of warps running compared to theoretical max.                |
| **Issue Slot Utilization**| Measures how often instruction issue slots are used – reflects ILP.       |
| **Branch Efficiency**     | Fraction of non-divergent branches in warp execution.                     |
| **Stall Reasons**         | Categorizes time lost to memory dependency, execution dependency, etc.    |

###  How to Use Nsight Compute

1. Launch profiling:
```bash
ncu --set full --target-processes all ./my_app
```
