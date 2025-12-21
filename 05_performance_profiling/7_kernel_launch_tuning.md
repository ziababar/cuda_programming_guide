#  Kernel Launch Config Tuning

**Previous: [Occupancy Analysis](6_occupancy_analysis.md)** | **Next: [Detecting Bottlenecks](8_detecting_bottlenecks.md)**

---

##  **Kernel Launch Config Tuning**

The configuration of your kernel launch — specifically the **grid** and **block** dimensions — has a significant impact on performance. CUDA does not auto-optimize launch parameters, so manual tuning is critical for peak efficiency.

---

###  Key Parameters

```cpp
kernel<<<numBlocks, threadsPerBlock>>>(...);
```

- `threadsPerBlock`: Controls how many threads run per block (typically 128–1024)
- `numBlocks`: Total number of blocks to launch = ceil(N / threadsPerBlock)

Choosing these wisely ensures full GPU utilization, optimal occupancy, and efficient memory access.

---

###  What to Experiment With

####  Varying Block Sizes

Try powers of two:
- 128 threads/block
- 256 threads/block (common sweet spot)
- 512 threads/block

 **Why it matters**:
- Impacts **occupancy** (how many warps/blocks can fit on an SM)
- Affects **coalescing patterns** in global memory
- Influences **register/shared memory pressure**

---

####  Tiling and Blocking

Use **tiling** in shared memory to divide data into reusable blocks.

Example:
```cpp
__shared__ float tileA[TILE_SIZE][TILE_SIZE];
__shared__ float tileB[TILE_SIZE][TILE_SIZE];

for (int k = 0; k < N/TILE_SIZE; ++k) {
  tileA[threadIdx.y][threadIdx.x] = A[row][k * TILE_SIZE + threadIdx.x];
  tileB[threadIdx.y][threadIdx.x] = B[k * TILE_SIZE + threadIdx.y][col];
  __syncthreads();
  // multiply tiles
  __syncthreads();
}
```

 **Why it matters**:
- Greatly improves **data locality**
- Reduces **global memory traffic**
- Enhances **shared memory reuse**

---

####  Loop Unrolling

Manually unroll inner loops when loop trip count is fixed and small.

```cpp
#pragma unroll
for (int i = 0; i < 4; i++) {
  sum += A[i] * B[i];
}
```

 **Why it matters**:
- Reduces loop control overhead
- Increases **Instruction-Level Parallelism (ILP)**
- Gives compiler more opportunity to optimize

---

###  Best Practices

- Use **`cudaOccupancyMaxPotentialBlockSize()`** to get a good starting point.
- Start with 256 or 512 threads per block and tune from there.
- Measure performance changes using **Nsight Compute** (especially SM Efficiency, Warp Execution Efficiency, and DRAM Throughput).
- Align block sizes with warp size (32) to avoid under-utilized warps.

---

###  Example: Matrix Multiplication

| Config                    | SM Efficiency | Occupancy | Speedup |
|---------------------------|---------------|-----------|---------|
| 1024 threads/block        | Low           | Low       | 1.0x    |
| 256 threads/block (tiled) | High          | High      | 3.8x    |
| 256 threads + unrolled    | Very High     | High      | 5.1x    |

---

Proper kernel launch tuning can turn a correct kernel into a **high-performance** one. It’s the bridge between working code and GPU-optimized code.
