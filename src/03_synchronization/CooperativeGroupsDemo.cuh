#ifndef COOPERATIVE_GROUPS_DEMO_CUH
#define COOPERATIVE_GROUPS_DEMO_CUH

#include <cooperative_groups.h>
#include <cstdio>
#include <cmath>

namespace cg = cooperative_groups;

namespace CooperativeGroupsDemo {

// Thread Block Group Example
__global__ void cooperative_block_reduction(float* data, float* output, int N) {
    cg::thread_block cta = cg::this_thread_block();
    int tid = cta.thread_rank();

    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < N) ? data[idx] : 0.0f;

    cta.sync();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        cta.sync();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Tiled Partition Example
__global__ void tiled_reduction(float* data, float* output) {
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    int tid = cta.thread_rank();
    float val = data[tid];

    for (int i = tile32.size() / 2; i > 0; i /= 2) {
        val += tile32.shfl_down(val, i);
    }

    if (tile32.thread_rank() == 0) {
        output[tid / 32] = val;
    }
}

// Grid Synchronization Example
__global__ void global_sync_kernel(float* data, int N) {
    cg::grid_group grid = cg::this_grid();
    int tid = grid.thread_rank();

    if (tid < N) {
        data[tid] = sqrt(data[tid]);
    }

    grid.sync();

    if (tid > 0 && tid < N - 1) {
        float left = data[tid - 1];
        float right = data[tid + 1];
        data[tid] = (left + right) * 0.5f;
    }
}

// Host helper
inline void launch_cooperative(float* d_data, int N) {
    int num_blocks = 32;
    int threads_per_block = 256;

    void* kernelArgs[] = { &d_data, &N };
    dim3 dimBlock(threads_per_block, 1, 1);
    dim3 dimGrid(num_blocks, 1, 1);

    cudaLaunchCooperativeKernel((void*)global_sync_kernel, dimGrid, dimBlock, kernelArgs);
}

} // namespace CooperativeGroupsDemo

#endif // COOPERATIVE_GROUPS_DEMO_CUH
