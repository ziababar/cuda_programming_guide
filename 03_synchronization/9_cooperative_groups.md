# Cooperative Groups

Cooperative Groups (introduced in CUDA 9.0) provides a flexible model for synchronization and communication within groups of threads. It decouples synchronization from the rigid thread block structure, allowing for safer and more composable parallel algorithms.

**Previous: [Advanced Patterns](8_advanced_synchronization.md)** | **Next: [Streams & Concurrency](../04_streams_concurrency/1_stream_fundamentals.md)**

---

## **Why Cooperative Groups?**

Legacy CUDA synchronization (`__syncthreads()`) is limited to the entire thread block. Cooperative Groups allows you to:
1.  **Define Flexible Groups**: Create groups smaller than a block (e.g., tiles) or larger (e.g., across the entire grid).
2.  **Safe Synchronization**: Prevents deadlocks caused by implicit assumptions about thread execution order.
3.  **Modular Code**: Pass group objects to functions, making them agnostic to the calling context (block vs. device).

To use Cooperative Groups, include the header:
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
```

---

## **Thread Block Groups**

The most basic group is the **Thread Block Group**. It replaces the implicit `blockIdx`, `threadIdx`, and `__syncthreads()`.

### **Basic Usage**

```cpp
__global__ void cooperative_block_reduction(float* data, float* output, int N) {
    // Get handle to the current thread block group
    cg::thread_block cta = cg::this_thread_block();

    // Group-specific thread index (equivalent to threadIdx.x)
    int tid = cta.thread_rank();

    // Shared memory for reduction
    extern __shared__ float sdata[];

    // Load data
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < N) ? data[idx] : 0.0f;

    // Synchronize the group (safer replacement for __syncthreads())
    cta.sync();

    // Reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // Sync only the participating threads in the group
        cta.sync();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

---

## **Tiled Partitions**

Tiled partitions break a thread block into smaller, independent sub-groups (tiles). This is highly efficient for warp-level operations and avoids the complexity of warp intrinsics (`__shfl`).

### **Powers of 2 Partitioning**

Commonly used sizes: 2, 4, 8, 16, 32.

```cpp
__global__ void tiled_reduction(float* data, float* output) {
    cg::thread_block cta = cg::this_thread_block();

    // Create a tile of 32 threads (a warp)
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    int tid = cta.thread_rank();
    float val = data[tid];

    // Parallel reduction using shuffle within the tile
    // No need for __shfl_down_sync, the tile abstraction handles it
    for (int i = tile32.size() / 2; i > 0; i /= 2) {
        val += tile32.shfl_down(val, i);
    }

    // The 0-th thread of the tile writes the result
    if (tile32.thread_rank() == 0) {
        output[tid / 32] = val;
    }
}
```

---

## **Grid Synchronization**

Standard CUDA limits synchronization to the thread block level. To synchronize **all threads in a grid**, you must use the **Grid Group**.

### **Requirements**
1.  **Kernel Launch**: Must use `cudaLaunchCooperativeKernel`.
2.  **Occupancy**: The GPU must be able to schedule *all* blocks simultaneously. If the grid is too large for the GPU's resident resources, the launch will fail.

### **Implementation**

```cpp
__global__ void global_sync_kernel(float* data, int N) {
    // Handle to the entire grid
    cg::grid_group grid = cg::this_grid();

    int tid = grid.thread_rank(); // Global thread ID

    // Phase 1: Compute
    if (tid < N) {
        data[tid] = sqrt(data[tid]);
    }

    // GLOBAL BARRIER: Synchronize all threads in the grid
    // This allows threads to safely read data written by ANY other thread
    grid.sync();

    // Phase 2: Compute using results from Phase 1
    // (e.g., neighbor averaging across block boundaries)
    if (tid > 0 && tid < N - 1) {
        float left = data[tid - 1];
        float right = data[tid + 1];
        data[tid] = (left + right) * 0.5f;
    }
}

// Host code
void launch_cooperative(float* d_data, int N) {
    int num_blocks = 32; // Must fit on device!
    int threads_per_block = 256;

    void* kernelArgs[] = { &d_data, &N };
    dim3 dimBlock(threads_per_block, 1, 1);
    dim3 dimGrid(num_blocks, 1, 1);

    cudaLaunchCooperativeKernel((void*)global_sync_kernel, dimGrid, dimBlock, kernelArgs);
}
```

---

## **Key Takeaways**

1.  **Safety**: Use `cg::this_thread_block().sync()` instead of `__syncthreads()` for clarity and future-proofing.
2.  **Flexibility**: Use `tiled_partition` for warp-level operations instead of raw `__shfl` intrinsics.
3.  **Global Sync**: Use `cg::this_grid().sync()` when you need to coordinate across the entire device, but be mindful of the `cudaLaunchCooperativeKernel` constraints.
4.  **Composition**: Groups can be passed to functions, making libraries easier to write.

## **Related Guides**

*   **Previous**: [Advanced Synchronization](8_advanced_synchronization.md)
*   **Next**: [Streams & Concurrency](../04_streams_concurrency/1_stream_fundamentals.md)
