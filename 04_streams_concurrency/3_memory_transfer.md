# Memory Transfer Optimization

Memory transfer optimization is crucial for achieving peak performance in CUDA applications. Understanding the memory hierarchy, transfer patterns, and bandwidth utilization strategies can significantly impact overall application throughput.

**[Back to Main CUDA Notes](../00_quick_start/0_cuda_cheat_sheet.md)** | **Previous: [Asynchronous Operations](2_asynchronous_operations.md)**

---

## Pinned Memory Deep Dive

Pinned (page-locked) memory is essential for achieving maximum memory transfer bandwidth and enabling true asynchronous operations. Standard `malloc` memory is pageable; the driver must copy it to a temporary pinned buffer before transferring to the GPU. `cudaHostAlloc` skips this step.

### Key Benefits
*   **Higher Bandwidth**: Avoids extra CPU copy.
*   **Asynchrony**: Allows `cudaMemcpyAsync` to return immediately.
*   **Mapped Memory**: Can be mapped into the device address space (Zero-Copy).

### PinnedMemoryManager Implementation

The `PinnedMemoryManager` class simplifies:
*   Allocation/Deallocation of pinned memory
*   Tracking memory usage
*   Managing allocation flags (WriteCombined, Mapped, etc.)

> **Full Implementation**: [`src/04_streams_concurrency/PinnedMemoryManager.cuh`](../src/04_streams_concurrency/PinnedMemoryManager.cuh)

```cpp
#include "../src/04_streams_concurrency/PinnedMemoryManager.cuh"

void demonstrate_pinned_memory() {
    PinnedMemoryManager manager;

    // Allocate write-combined memory (fast for CPU write, PCI-E read)
    void* ptr = manager.allocate(1024*1024, cudaHostAllocWriteCombined);

    // Use memory...

    manager.deallocate(ptr);
}
```

## Bandwidth Optimization Strategies

Maximizing saturation of the PCIe bus requires careful tuning of transfer sizes and concurrency.

### Optimization Techniques
1.  **Batching**: Group small transfers into larger chunks.
2.  **Concurrency**: Use multiple streams to saturate the link (though one stream often saturates PCIe x16).
3.  **Direction**: Bi-directional transfers can utilize full duplex PCIe bandwidth.

### BandwidthOptimizer Implementation

The `BandwidthOptimizer` helps identify optimal parameters:
*   Test transfer sizes
*   Test concurrent stream counts

> **Full Implementation**: [`src/04_streams_concurrency/BandwidthOptimizer.cuh`](../src/04_streams_concurrency/BandwidthOptimizer.cuh)

## Bidirectional Transfer Patterns

PCIe is full-duplex, meaning it can send and receive simultaneously. Overlapping Host-to-Device (H2D) and Device-to-Host (D2H) copies can double effective throughput.

### BidirectionalTransferManager Implementation

Manages dual-ring buffers to sustain simultaneous traffic in both directions.

> **Full Implementation**: [`src/04_streams_concurrency/BidirectionalTransferManager.cuh`](../src/04_streams_concurrency/BidirectionalTransferManager.cuh)

```cpp
#include "../src/04_streams_concurrency/BidirectionalTransferManager.cuh"

void demonstrate_bidirectional() {
    BidirectionalTransferManager manager(4, 1024*1024);
    manager.demonstrate_bidirectional_overlap();
}
```
