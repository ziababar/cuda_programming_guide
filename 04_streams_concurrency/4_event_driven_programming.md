# Event-Driven Programming

CUDA events provide precise synchronization control and performance measurement capabilities, enabling sophisticated coordination between streams and accurate timing analysis of GPU operations.

## Event Fundamentals and Types

CUDA events are markers recorded in a stream. They can be used for:
1. **Synchronization**: Waiting for an event to complete.
2. **Timing**: Measuring elapsed time between two events.

### Event Types
- **`cudaEventDefault`**: Standard event with timing support.
- **`cudaEventBlockingSync`**: Host thread yields while waiting (lower CPU usage, potential latency).
- **`cudaEventDisableTiming`**: Optimized for synchronization only (slightly lower overhead).
- **`cudaEventInterprocess`**: For sharing events across processes.

> **Implementation**: An `EventManager` class wrapping these functionalities is provided in [`src/04_streams_concurrency/event_manager.cuh`](../src/04_streams_concurrency/event_manager.cuh).

## Precision Timing and Performance Measurement

Events offer the most accurate way to time GPU operations, avoiding the pitfalls of host-side timers (which include driver overhead and latency).

### Performance Profiling
A `PerformanceProfiler` class ([`src/04_streams_concurrency/performance_profiler.cuh`](../src/04_streams_concurrency/performance_profiler.cuh)) allows you to:
- Define timing regions using RAII (Resource Acquisition Is Initialization).
- Collect statistics (min, max, average, std dev).
- Benchmark specific operations.

```cpp
// Example: Using the profiler
PerformanceProfiler profiler;
{
    auto timer = profiler.time_scope("my_kernel");
    my_kernel<<<...>>>();
}
profiler.print_all_stats();
```

## Advanced Synchronization Patterns

Events enable complex dependency graphs where streams wait on each other without host intervention.

### Event Coordinator
The `EventCoordinator` class ([`src/04_streams_concurrency/event_coordinator.cuh`](../src/04_streams_concurrency/event_coordinator.cuh)) demonstrates how to build and execute a dependency graph of tasks.
- **Nodes**: Work units (kernels).
- **Dependencies**: Events that must complete before a node starts.
- **Execution**: The coordinator schedules nodes onto streams as their dependencies are satisfied.
