# CUDA Streams & Concurrency

CUDA streams are the backbone of high-performance GPU programming, enabling asynchronous execution, memory transfer overlap, and sophisticated pipeline orchestration. This chapter provides a comprehensive guide to mastering concurrency in CUDA.

## Table of Contents

1. **[Stream Fundamentals](1_stream_fundamentals.md)**
   - Stream Types and Properties
   - Stream Execution Model
   - Stream Management Patterns

2. **[Asynchronous Operations](2_asynchronous_operations.md)**
   - Compute-Transfer Overlap
   - Stream Synchronization Mechanisms
   - Dynamic Stream Management

3. **[Memory Transfer Optimization](3_memory_transfer.md)**
   - Pinned Memory Deep Dive
   - Bandwidth Optimization Strategies
   - Advanced Transfer Patterns

4. **[Event-Driven Programming](4_event_driven_programming.md)**
   - Event Fundamentals and Types
   - Precision Timing and Performance Measurement
   - Advanced Synchronization Patterns

5. **[CUDA Graphs Deep Dive](5_cuda_graphs.md)**
   - Graph Fundamentals and Architecture
   - Advanced Graph Patterns and Optimization
   - Production Graph Optimization

6. **[Advanced Patterns](6_advanced_patterns.md)**
   - Producer-Consumer Patterns
   - Pipeline Architecture Patterns
   - Dynamic Load Balancing

## Source Code

The implementations discussed in this chapter are available in the `src/04_streams_concurrency/` directory. These include production-ready classes for managing streams, events, and graphs.

- `stream_manager.cuh`: Advanced stream pooling and priority management.
- `stream_sync.cuh`: Synchronization primitives and patterns.
- `adaptive_stream_manager.cuh`: Dynamic stream allocation.
- `pinned_memory_manager.cuh`: Robust pinned memory handling.
- `bandwidth_optimizer.cuh`: Tools for optimizing transfer sizes.
- `bidirectional_transfer.cuh`: Managing full-duplex transfers.
- `event_manager.cuh`: Wrapper for CUDA events.
- `performance_profiler.cuh`: High-precision GPU timing.
- `event_coordinator.cuh`: Dependency graph execution using events.
- `graph_manager.cuh`: Comprehensive CUDA Graph management.
- `graph_patterns.cuh`: Advanced graph usage patterns.
- `graph_optimizer.cuh`: Optimization strategies for graphs.
- `producer_consumer.cuh`: Multi-buffered producer-consumer system.
- `stream_pipeline.cuh`: Multi-stage processing pipelines.
- `stream_balancer.cuh`: Dynamic load balancing across streams.
