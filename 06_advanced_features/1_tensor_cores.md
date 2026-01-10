# Tensor Cores and WMMA

NVIDIA Tensor Cores are specialized hardware units designed to accelerate matrix operations, particularly those found in deep learning workloads.

## Key Concepts

- **Mixed Precision**: Perform matrix multiplication in FP16/BF16 while accumulating in FP32.
- **WMMA API**: Warp Matrix Multiply Accumulate (C++ API).
- **Fragments**: Register-based data structures holding matrix tiles.

## WMMA API Workflow

1.  **Load**: Load matrix fragments from memory to registers (`wmma::load_matrix_sync`).
2.  **Compute**: Perform matrix multiplication (`wmma::mma_sync`).
3.  **Store**: Store result fragments back to memory (`wmma::store_matrix_sync`).

## Code Example

See `src/06_advanced_features/TensorCoreDemo.cuh` for a complete implementation.

```cpp
#include <mma.h>
using namespace nvcuda;

// Dimensions: 16x16x16
__global__ void wmma_kernel(half *a, half *b, float *c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```

## Performance Tips

- **Data Layout**: Ensure data is in a layout compatible with Tensor Cores (e.g., NHWC for convolutions).
- **Warp Alignment**: WMMA operations are warp-synchronous; all threads in a warp must participate.
