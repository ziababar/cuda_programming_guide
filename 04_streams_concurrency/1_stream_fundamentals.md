# Stream Fundamentals

CUDA streams represent ordered sequences of GPU operations that execute asynchronously with respect to the host and other streams, enabling sophisticated concurrency patterns.

**[Back to Streams & Concurrency Index](1_cuda_streams_concurrency.md)**

---

## Stream Types and Properties

### Stream Hierarchy and Characteristics

The following code demonstrates the fundamentals of CUDA streams, including default stream behavior, explicit stream creation, and priority configuration.

**Code Example:** [`stream_fundamentals.cuh`](../src/04_streams_concurrency/stream_fundamentals.cuh)

### Stream Execution Model

Streams allow for FIFO ordering within a single stream, while enabling concurrency between different streams.

**Code Example:** [`stream_execution.cuh`](../src/04_streams_concurrency/stream_execution.cuh)

### Stream Management Patterns

Advanced stream management involves handling multiple streams with different priorities and assigning them to appropriate workloads.

**Code Example:** [`StreamManager.cuh`](../src/04_streams_concurrency/StreamManager.cuh)

---

## Multi-GPU Coordination

While this guide focuses on single-GPU stream concurrency, multi-GPU applications leverage streams to manage work across devices.

*   **One Context per Thread (Typical):**  Usually, a host thread manages one GPU context. To control multiple GPUs, you can use multiple host threads, or switch contexts using `cudaSetDevice()`.
*   **Streams per Device:** Each device has its own default stream and can have its own created streams.
*   **Peer-to-Peer Access:** `cudaDeviceEnablePeerAccess` allows kernels on one GPU to access memory on another.
*   **Synchronization:** `cudaStreamWaitEvent` can be used to synchronize streams across devices (if the event is created with `cudaEventInterprocess` or used within the same process context).

*(Note: A comprehensive Multi-GPU guide is planned for a future update.)*

---

## Nsight Debugging Tips

- Use **Nsight Systems** to visualize:
  - Stream timelines
  - Overlap of memcopy and kernels
- Identify serialization caused by:
  - Shared resources
  - Host sync calls (`cudaDeviceSynchronize()`)
