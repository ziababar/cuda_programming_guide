# Dynamic Parallelism

Dynamic Parallelism allows CUDA kernels to launch new kernels directly from the GPU, without involving the CPU.

## Use Cases

- **Adaptive Refinement**: Dynamically spawn more threads for complex regions of a dataset.
- **Recursive Algorithms**: Implement recursion where each step launches parallel work.
- **Graph Traversal**: Dynamically explore nodes without CPU round-trips.

## Implementation

See `src/06_advanced_features/DynamicParallelismDemo.cuh`.

```cpp
__global__ void child_kernel() {
    // Child work
}

__global__ void parent_kernel() {
    if (threadIdx.x == 0) {
        // Launch child from device
        child_kernel<<<1, 32>>>();

        // Optional: Wait for child
        cudaDeviceSynchronize();
    }
}
```

## Compilation

Dynamic Parallelism requires Relocatable Device Code (`-rdc=true`).

```bash
nvcc -arch=sm_70 -rdc=true main.cu -o app
```
