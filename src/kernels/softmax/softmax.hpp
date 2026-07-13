#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ======================================================
// Llama 3.2 1B Constants
// ======================================================

constexpr int SEQ_LEN = 512;
constexpr int HEADS = 32;
constexpr int KV_HEADS = 8;
constexpr int HEAD_DIM = 64;

constexpr int BLOCK_SIZE = 256;
constexpr int WARPS = 8;
constexpr int WARP_SIZE = 32;

constexpr int VALUES_PER_THREAD = 2;
constexpr int ELEMENTS_PER_THREAD = SEQ_LEN / BLOCK_SIZE;

constexpr unsigned FULL_MASK = 0xffffffffu;

// ======================================================
// Kernel
// ======================================================

__global__ void softmax_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int rows
);

// ======================================================
// Launcher
// ======================================================

cudaError_t launch_softmax_safe(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int rows
);

#endif // SOFTMAX_HPP