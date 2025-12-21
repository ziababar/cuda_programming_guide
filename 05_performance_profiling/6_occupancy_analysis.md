#  Occupancy Analysis

**Previous: [Typical Optimization Strategies](5_optimization_strategies.md)** | **Next: [Kernel Launch Config Tuning](7_kernel_launch_tuning.md)**

---

##  **Occupancy Analysis**

**Occupancy** refers to the ratio of **active warps per Streaming Multiprocessor (SM)** to the **maximum possible warps** on that architecture. It provides a rough measure of how effectively the GPU hardware is being utilized.

High occupancy generally improves performance by allowing the GPU to **hide memory latency** and maintain throughput — but **higher is not always better**.


###  API Usage

To estimate optimal configuration:

```cpp
int blockSize;   // Suggested threads per block
int minGridSize; // Minimum grid size to achieve full occupancy

cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel);
```

To calculate active blocks per SM given your parameters:

```cpp
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, myKernel, threadsPerBlock, sharedMemPerBlock);
```

 Factors That Influence Occupancy
| Factor	| Description |
|-------------------------|---------------------------------------------|
| Threads per Block	| Larger blocks allow more warps but may exceed register or shared memory limits |
| Registers per Thread	| High register use may limit number of active warps |
| Shared Memory per Block	| Using too much shared memory reduces block concurrency |
| Warp Scheduling Limits	| SMs have hardware limits (e.g., 64 warps, 2048 threads, 16 blocks) |

 Common Pitfall
Too high occupancy may:
- Cause register spilling (to local/global memory)
- Increase contention for shared memory or caches
- Limit instruction-level parallelism (ILP)

 Best Practices
- Aim for balanced occupancy (50–80%) unless your kernel is extremely latency-bound.
- Use --ptxas-options=-v with nvcc to inspect register and shared memory usage:

nvcc -arch=sm_80 --ptxas-options=-v mykernel.cu

- Profile with Nsight Compute:
    - Look at Achieved Occupancy
    - Compare performance across different launch configurations

 Real-World Example
Matrix multiplication kernel:
- Initial config: 512 threads/block → High register usage → Only 2 blocks per SM.
- Tuning: Reduced threads to 256/block, reduced register pressure via compiler flags.
- Result: Occupancy improved from 37% to 75%, leading to a 2.1x speedup.

Occupancy is not the only factor that determines kernel performance, but it is a powerful lever to tune concurrency and throughput on the GPU.
