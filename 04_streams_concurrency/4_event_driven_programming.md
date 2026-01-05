# Event-Driven Programming

CUDA events provide precise synchronization control and performance measurement capabilities, enabling sophisticated coordination between streams and accurate timing analysis of GPU operations.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Previous: [Memory Transfer](3_memory_transfer.md)**

---

## Event Fundamentals

CUDA events serve as lightweight synchronization primitives that can be recorded in streams and waited upon.

### Event Capabilities
1.  **Synchronization**: `cudaStreamWaitEvent` allows a stream to wait for an event recorded in another stream without blocking the host.
2.  **Timing**: `cudaEventElapsedTime` measures time between two recorded events with high precision.
3.  **Host Sync**: `cudaEventSynchronize` blocks the CPU until the event is recorded.

### EventManager Implementation

A robust wrapper for CUDA event lifecycle and common operations.

> **Full Implementation**: [`src/04_streams_concurrency/EventManager.cuh`](../src/04_streams_concurrency/EventManager.cuh)

```cpp
#include "../src/04_streams_concurrency/EventManager.cuh"

void demonstrate_events() {
    EventManager manager;

    // Create blocking sync event (low CPU usage)
    int evt_id = manager.create_event("sync_point", cudaEventBlockingSync);

    // Record in stream
    manager.record_event("sync_point", stream);

    // Wait on host
    manager.synchronize_event("sync_point");
}
```

## Precision Timing and Profiling

Events are the most accurate way to measure kernel execution time, avoiding host-side OS jitter.

### PerformanceProfiler Implementation

A scoped, RAII-based profiler using CUDA events.

> **Full Implementation**: [`src/04_streams_concurrency/PerformanceProfiler.cuh`](../src/04_streams_concurrency/PerformanceProfiler.cuh)

```cpp
#include "../src/04_streams_concurrency/PerformanceProfiler.cuh"

void profile_code() {
    PerformanceProfiler profiler;

    {
        auto guard = profiler.time_scope("my_kernel");
        my_kernel<<<...>>>();
    } // Timer stops automatically here

    profiler.print_all_stats();
}
```

## Advanced Synchronization Patterns

Events enable complex dependency graphs (DAGs) without using CUDA Graphs, which is useful for dynamic dependencies.

### EventCoordinator Implementation

Manages a DAG of tasks where execution is triggered by event completion.

> **Full Implementation**: [`src/04_streams_concurrency/EventCoordinator.cuh`](../src/04_streams_concurrency/EventCoordinator.cuh)

```cpp
#include "../src/04_streams_concurrency/EventCoordinator.cuh"

void dependency_graph_demo() {
    // ... setup streams ...
    EventCoordinator coordinator(streams);

    // Define dependencies: Stage 2 depends on Stage 1
    coordinator.add_node("stage1", {}, func1);
    coordinator.add_node("stage2", {"stage1"}, func2);

    coordinator.execute_graph();
}
```
