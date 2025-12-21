#  CUDA Profiling & Optimization – Deep Dive

## 1.  Key Profiling Tools

| Tool               | Description                                                                                          | Use Cases                                                                 |
|--------------------|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **Nsight Systems** | System-wide visualization and tracing of CPU-GPU activity. Provides a **timeline view** across host, device, and memory. | Identify stream overlaps, asynchronous kernel behavior, data transfer bottlenecks. |
| **Nsight Compute** | Deep dive into per-kernel execution with metrics like **SM utilization**, **warp efficiency**, **memory throughput**. | Analyze low-level performance issues inside individual kernels.           |
| `nvprof`           | Lightweight command-line profiler. Deprecated in favor of Nsight tools but useful for quick checks. | Fast profiling when you want to collect kernel timings or memory stats with minimal overhead. |
| `cuda-memcheck`    | Runtime memory error checker. Detects **out-of-bounds**, **race conditions**, **invalid memory accesses**. | Validate correctness before optimization; critical for shared/global memory debugging. |
| `cuda-gdb`         | Source-level debugger for device and host code. Supports breakpoints, watchpoints, and step-through. | Debug control flow issues, conditional logic, and thread-local behavior. |

---

###  Tips for Tool Usage

-  **Start with Nsight Systems** to find **where** bottlenecks occur (e.g. CPU wait time, GPU idle).
-  Use **Nsight Compute** to find **why** a kernel is underperforming (e.g. low occupancy, warp divergence).
-  Always run **cuda-memcheck** before tuning for performance — correctness comes first.
-  Use **cuda-gdb** to diagnose crashes or logic errors in device code when print-style debugging fails.
