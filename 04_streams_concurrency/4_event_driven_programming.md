# Event-Driven Programming

CUDA events provide precise synchronization control and performance measurement capabilities, enabling sophisticated coordination between streams and accurate timing analysis of GPU operations.

## Event Fundamentals and Types

CUDA events serve as lightweight synchronization primitives.

### Comprehensive Event Management

The `EventManager` class simplifies creating, recording, and synchronizing events with different flags (Default, BlockingSync, DisableTiming, Interprocess).

**Source Code**: [`../src/04_streams_concurrency/event_manager.cuh`](../src/04_streams_concurrency/event_manager.cuh)

## Precision Timing and Performance Measurement

Events provide the most accurate method for measuring GPU execution times.

### Advanced Timing Infrastructure

The `PerformanceProfiler` class uses events to measure execution time of regions, supports RAII-style timing guards, and statistical analysis.

**Source Code**: [`../src/04_streams_concurrency/performance_profiler.cuh`](../src/04_streams_concurrency/performance_profiler.cuh)

## Advanced Synchronization Patterns

### Event-Based Coordination Patterns

The `EventCoordinator` class builds complex dependency graphs using events to coordinate execution across multiple streams.

**Source Code**: [`../src/04_streams_concurrency/event_coordinator.cuh`](../src/04_streams_concurrency/event_coordinator.cuh)
