#ifndef TENSOR_CORE_DEMO_CUH
#define TENSOR_CORE_DEMO_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <vector>

using namespace nvcuda;

// Matrix dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Tensor Core Kernel for 16x16x16 Matrix Multiplication
// C = A * B + C
// A, B: half precision (fp16)
// C: single precision (fp32)
__global__ void wmma_example(half *a, half *b, float *c, int lda, int ldb, int ldc) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, lda);
    wmma::load_matrix_sync(b_frag, b, ldb);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, ldc, wmma::mem_row_major);
}

class TensorCoreDemo {
public:
    static void run_demo() {
        printf("=== Tensor Core (WMMA) Demo ===\n");

        int lda = WMMA_K;
        int ldb = WMMA_K;
        int ldc = WMMA_N;

        // Host memory
        std::vector<half> h_a(WMMA_M * WMMA_K);
        std::vector<half> h_b(WMMA_K * WMMA_N);
        std::vector<float> h_c(WMMA_M * WMMA_N);

        // Initialize with data
        for (int i = 0; i < h_a.size(); ++i) h_a[i] = __float2half(1.0f);
        for (int i = 0; i < h_b.size(); ++i) h_b[i] = __float2half(1.0f);

        // Device memory
        half *d_a, *d_b;
        float *d_c;
        cudaMalloc(&d_a, h_a.size() * sizeof(half));
        cudaMalloc(&d_b, h_b.size() * sizeof(half));
        cudaMalloc(&d_c, h_c.size() * sizeof(float));

        cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(half), cudaMemcpyHostToDevice);

        // Launch kernel with one warp (32 threads)
        wmma_example<<<1, 32>>>(d_a, d_b, d_c, lda, ldb, ldc);
        cudaDeviceSynchronize();

        // Check errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        } else {
            cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost);
            printf("WMMA 16x16x16 multiplication successful.\n");
            printf("Top-left element: %f (Expected: 16.0)\n", h_c[0]);
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
};

#endif // TENSOR_CORE_DEMO_CUH
