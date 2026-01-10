#ifndef COOPERATIVE_GROUPS_DEMO_CUH
#define COOPERATIVE_GROUPS_DEMO_CUH

#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Demonstrate Thread Block Group
__global__ void thread_block_group_kernel() {
    cg::thread_block tb = cg::this_thread_block();

    // Synchronize all threads in the block
    tb.sync();

    if (tb.thread_rank() == 0) {
        printf("Thread Block Group: Block %d, Thread %d (Rank 0)\n",
               blockIdx.x, threadIdx.x);
        printf("  Group Size: %lld\n", tb.size());
    }

    tb.sync();
}

// Demonstrate Tiled Partition
__global__ void tiled_partition_demo_kernel(int* data, int size) {
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(tb);

    int tid = tb.thread_rank();

    if (tid < size) {
        int val = data[tid];

        // Parallel reduction within a warp (tile)
        for (int i = tile32.size() / 2; i > 0; i /= 2) {
            val += tile32.shfl_down(val, i);
        }

        if (tile32.thread_rank() == 0) {
            printf("Warp %d reduction result: %d\n", tid / 32, val);
        }
    }
}

class CooperativeGroupsDemo {
public:
    static void run_demos() {
        printf("=== Cooperative Groups Demo ===\n");

        printf("\n1. Thread Block Group:\n");
        thread_block_group_kernel<<<2, 64>>>();
        cudaDeviceSynchronize();

        printf("\n2. Tiled Partition (Warp Reduction):\n");
        const int size = 256;
        std::vector<int> h_data(size, 1); // Fill with 1s
        int* d_data;
        cudaMalloc(&d_data, size * sizeof(int));
        cudaMemcpy(d_data, h_data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

        tiled_partition_demo_kernel<<<1, size>>>(d_data, size);
        cudaDeviceSynchronize();

        cudaFree(d_data);
    }
};

#endif // COOPERATIVE_GROUPS_DEMO_CUH
