#  CUDA Streams & Concurrency Complete Guide

CUDA streams are the backbone of high-performance GPU programming, enabling asynchronous execution, memory transfer overlap, and sophisticated pipeline orchestration. Understanding streams deeply is essential for achieving optimal GPU utilization and building scalable parallel applications.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Related: [Memory Hierarchy](../02_memory_hierarchy/1_cuda_memory_hierarchy.md)** | **Architecture: [Execution Model](../01_execution_model/1_cuda_execution_model.md)**

---

##  **Table of Contents**

1. [Stream Fundamentals](1_stream_fundamentals.md)
2. [Asynchronous Operations](2_asynchronous_operations.md)
3. [Memory Transfer Optimization](3_memory_transfer.md)
4. [Event-Driven Programming](4_event_driven_programming.md)
5. [CUDA Graphs Deep Dive](5_cuda_graphs.md)
6. [Advanced Stream Patterns](6_advanced_patterns.md)

---

##  **Quick Reference**

### **Stream Hierarchy:**
```
Host Application
 Default Stream (Blocking)
 Explicit Streams (Async)
    Memory Transfers
    Kernel Executions
    Event Synchronization
 CUDA Graphs (Static DAG)
     Node Dependencies
     Optimized Execution
```

### **Key Performance Concepts:**
| Concept | Description | Performance Impact |
|---------|-------------|-------------------|
| **Stream Overlap** | Concurrent compute + memory transfer | 2-4x throughput improvement |
| **Pinned Memory** | Host memory accessible by DMA | 2-3x transfer speed |
| **Event Synchronization** | Fine-grained stream coordination | Minimal overhead |
| **CUDA Graphs** | Static execution DAG | 50-90% launch overhead reduction |
| **Multi-Stream** | Parallel execution contexts | Near-linear scaling |

---

For detailed documentation and code examples, please refer to the individual sections listed in the Table of Contents above. Each section contains comprehensive guides and production-ready code implementations in `src/04_streams_concurrency/`.
