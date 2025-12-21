#  Nsight Systems – Timeline Debugging

**Previous: [Profiling Overview](1_profiling_overview.md)** | **Next: [Nsight Compute](3_nsight_compute.md)**

---

##  **Nsight Systems – Timeline Debugging**

`Nsight Systems` provides a **top-down, system-wide view** of your application. It captures CPU, GPU, memory, and OS interactions in a **timeline-based UI**, making it ideal for analyzing bottlenecks related to synchronization, concurrency, and data transfers.


###  Use Cases

-  **Stream Concurrency**
  Visualize how multiple CUDA streams overlap. Helps you confirm whether your compute and memory operations run in parallel or serialize unnecessarily.

-  **Serialization Detection**
  Detect if memory copies and kernels are serialized on the same stream due to dependencies or poor stream usage.

-  **CPU–GPU Sync Bottlenecks**
  Identify `cudaMemcpy` or `cudaDeviceSynchronize()` calls that block CPU threads while waiting on the GPU. Look for gaps between kernel launches and compute.

-  **Multi-GPU Workload Analysis**
  Evaluate load balancing across GPUs in multi-GPU systems. See whether each GPU is fully utilized and identify synchronization overhead.


###  Key Metrics & Events

| Metric/Event             | What It Tells You                                                    |
|--------------------------|----------------------------------------------------------------------|
| **Kernel Launch Time**    | When and how long each kernel runs. Look for launch delays.          |
| **Memcpy Overlap**        | Whether memory transfers overlap with computation (for async copies).|
| **Host Thread Blocking**  | CPU-side stalls due to synchronization or memory transfers.          |
| **Stream Usage Timeline** | Which streams are active/inactive and how they overlap.              |
| **CPU–GPU Correlation**   | Track dependencies between CPU launches and GPU execution.           |
| **NVTX Markers**          | Annotate timeline with custom labels (e.g. per frame, phase, step).  |


###  Best Practices

- Use **CUDA events and NVTX ranges** to annotate key sections of your code. This enhances timeline readability and correlates logical phases with low-level execution.

- Confirm that memory transfers (e.g. `cudaMemcpyAsync`) and kernel executions on different streams **overlap as expected**.

- Check for long **idle gaps** between GPU operations — this usually signals a CPU bottleneck or missing prefetching/synchronization error.


###  Example Workflow

1. Launch your application with:
```bash
nsys profile --trace=cuda,nvtx,osrt ./my_app
```
