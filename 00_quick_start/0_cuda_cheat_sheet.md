# CUDA Technical Mastery – Cheat Sheet


## 1.  GPU Architecture

| Concept | Description | Key Notes |
|---------|-------------|-----------|
| **GPU** | Graphics Processing Unit with many parallel cores | Contains multiple SMs (e.g., A100 has 108 SMs) |
| **SM** | Streaming Multiprocessor - execution unit | Contains CUDA cores, shared mem, registers |
| **CUDA Core** | Basic arithmetic unit within SM | Executes individual thread instructions |
| **Warp Scheduler** | Hardware that manages warp execution | 4 warp schedulers per SM (typical) |
| **Memory Controllers** | Manage access to global memory | Determine memory bandwidth |
| **Tensor Cores** | Specialized units for AI workloads | Mixed-precision matrix operations |

 **Best Practice**: Choose GPU based on workload requirements (compute vs memory intensive).

###  **Hardware Hierarchy**

![GPU Architecture](../images/gpu_architecture.png)

**Key Specifications (Example: A100)**:
- **SMs**: 108 Streaming Multiprocessors
- **CUDA Cores**: 64 per SM (6,912 total)
- **Tensor Cores**: 4 per SM (432 total)
- **Memory**: 40-80GB HBM2e
- **Bandwidth**: 1.6TB/s memory bandwidth

**Architecture Impact**:
- **More SMs** = More parallel blocks
- **More CUDA cores per SM** = More threads per block
- **Higher memory bandwidth** = Better data throughput
- **Tensor cores** = Accelerated AI/ML workloads




## 2.  Execution Model

| Concept           | Description                                           | Key Notes                                             |
|------------------|-------------------------------------------------------|--------------------------------------------------------|
| **Thread**        | Basic unit of execution                              | Executes kernel code on CUDA cores                   |
| **Warp**          | Group of 32 threads executed together                | Warp divergence hurts performance                     |
| **Block**         | Group of warps assigned to an SM                     | Shares **shared memory**, can sync via `__syncthreads()` |
| **Grid**          | Collection of blocks distributed across SMs          | Can be 1D, 2D, 3D                                     |
| **Indexing**      | `threadIdx`, `blockIdx`, `blockDim`, `gridDim`       | Used to access per-thread data                        |

 **Best Practice**: Align your kernel design to problem geometry (e.g. 2D grid for images).

```
         Grid (Kernel Launch)
                  |
        +---------+---------+
        |                   |
    Block (0,0)         Block (1,0)  ...
        |                   |
   +----+----+         +----+----+
   |         |         |         |
Thread(0,0) Thread(1,0) ...     ...
```

**Execution Flow:**
1. **Kernel Launch**: Host launches a grid of thread blocks
2. **Block Distribution**: CUDA runtime distributes blocks across available SMs
3. **Warp Scheduling**: Each SM schedules warps (groups of 32 threads) for execution
4. **Thread Execution**: Individual threads execute kernel code in parallel

**Key Relationships:**
- Multiple **blocks** can run on a single **SM** (if resources allow)
- Each **block** is divided into **warps** of 32 consecutive threads
- All threads in a **warp** execute the same instruction (SIMT model)
- **Blocks** within a **grid** can execute in any order (independence requirement)

###  **Kernel Fundamentals**
| Concept | Description | Key Notes |
|---------|-------------|-----------|
| **Kernel** | `__global__` function executed by all threads in parallel | Uses `<<<grid, block>>>` launch syntax |
| **Launch Config** | `kernel<<<blocks, threads>>>(args)` | Defines grid and block dimensions |
| **Thread Index** | Each thread calculates unique ID for data access | `int id = blockIdx.x * blockDim.x + threadIdx.x` |

**Example**:
```cpp
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
// Launch: vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
```



## 3.  Memory Hierarchy

| Type            | Scope     | Speed     | Use Case                                     |
|-----------------|-----------|-----------|----------------------------------------------|
| **Registers**    | Thread    | Fastest   | Scalar/local variables                        |
| **Shared Mem**   | Block     | Very Fast | Thread collaboration within block             |
| **Global Mem**   | Grid      | Slow      | Main GPU memory                               |
| **Constant Mem** | Grid      | Fast (RO) | Broadcast constants to all threads            |
| **Texture/Surface** | Grid   | Optimized | 2D spatial locality, read-only                |
| **Unified Mem**  | Grid/Host | Medium    | Simplified allocation, slower for frequent access |

 **Best Practice**:
- Coalesce global memory access
- Use shared memory to reduce global loads
- Avoid shared memory **bank conflicts**


```
      Registers (Thread Private)
                 |
    Shared Memory / L1 Cache (Block Local)
                 |
      L2 Cache (Global, All SMs)
                 |
      Global Memory (Device DRAM)
                 |
      System Memory (Host RAM)
```

**Memory Access Patterns:**
1. **Registers**: Fastest access, private to each thread, limited capacity (~64KB per SM)
2. **Shared Memory**: Fast inter-thread communication within a block, ~48-164KB per SM
3. **L1/L2 Cache**: Hardware-managed caching of global memory accesses
4. **Global Memory**: Main GPU DRAM, highest capacity but slowest access
5. **Constant Memory**: Read-only, cached, excellent for broadcasting small data
6. **Texture Memory**: Optimized for 2D spatial locality with built-in filtering

**Performance Guidelines:**
- **Maximize register usage** for frequently accessed variables
- **Use shared memory** for data sharing between threads in a block
- **Ensure coalesced access** to global memory (consecutive threads access consecutive addresses)
- **Minimize global memory traffic** by reusing data through shared memory or registers



## 4.  Streams & Concurrency

| Concept              | Description                                              |
|----------------------|----------------------------------------------------------|
| **Streams**           | Command queues for async operations                      |
| **Default stream**    | Implicitly synchronizes                                  |
| **Multiple streams**  | Enable overlapping memcpy & compute                      |
| **Stream priorities** | Control execution ordering                               |
| **CUDA Graphs**       | Capture & replay workflows with less overhead            |

 **Best Practice**:
- Use pinned host memory (`cudaHostAlloc`) for async memcpy
- Use `cudaMemcpyAsync()` and multiple streams for throughput

---

## 5.  Profiling & Optimization

| Tool               | Use Case                             |
|--------------------|--------------------------------------|
| `nvprof`           | Basic kernel and memory profiling    |
| **Nsight Compute** | Kernel-level performance analysis    |
| **Nsight Systems** | Full system profiling (timeline)     |
| `cuda-gdb`         | Debugger for CUDA code               |

###  Optimization Metrics
- **Occupancy**: Threads/SM efficiency
- **Memory Throughput**: Avoid serialization
- **Warp Efficiency**: Avoid divergence
- **Execution Dependencies**: Hide latency with async

 **Best Practice**:
- Balance compute vs memory (Roofline model)
- Use shared memory tiling for matrix ops

---

## 6.  Advanced Features

| Feature              | Description                                      |
|----------------------|--------------------------------------------------|
| **Cooperative Groups**| Fine-grained synchronization within warps/groups|
| **Dynamic Parallelism**| Launch kernels from within kernels              |
| **Tensor Cores**      | Fused matrix-multiply for DL (use `mma.sync`)   |
| **cuBLAS/cuDNN/cuFFT**| Optimized CUDA libraries                        |

 **Best Practice**:
- Use libraries when possible over hand-coded kernels
- Use `cudaFuncAttributes` to tune resource use

---

## 7.  Multi-GPU & Unified Memory

| Concept               | Details                                            |
|------------------------|----------------------------------------------------|
| **P2P Access**         | Direct access between GPUs via NVLink              |
| **cudaMemcpyPeer**     | Fast inter-GPU data transfer                       |
| **Unified Memory**     | `cudaMallocManaged` for simplified pointer mgmt    |
| **Prefetching**        | Use `cudaMemPrefetchAsync()` for hinting data loc  |

 **Best Practice**:
- Avoid relying solely on Unified Memory for perf-sensitive workloads
- Enable ECC for mission-critical enterprise workloads

---

##  Key Code Snippets

```cpp
// Thread Indexing (2D Grid)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

// Shared Memory Usage
__shared__ float tile[32][32];
tile[threadIdx.y][threadIdx.x] = input[y][x];
__syncthreads();

// Async Copy
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```
  |