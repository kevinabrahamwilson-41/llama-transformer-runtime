#pragma once

// ======================================================
// Elementwise Multiply
// Llama 3.2 1B
// ======================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

// ======================================================
// Model Constants
// ======================================================

constexpr int SEQ_LEN    = 512;
constexpr int FFN_DIM    = 8192;
constexpr int BLOCK_SIZE = 256;

// ======================================================
// CUDA Error Checking
// ======================================================

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess)                                   \
    {                                                         \
        printf("CUDA Error %s:%d : %s\n",                     \
               __FILE__,                                      \
               __LINE__,                                      \
               cudaGetErrorString(err));                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

// ======================================================
// Kernel
// ======================================================

__global__
void elementwise_mul_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ output
);

// ======================================================
// Launcher
// ======================================================

void launch_elementwise_mul(
    const __nv_bfloat16* d_a,
    const __nv_bfloat16* d_b,
    __nv_bfloat16* d_output,
    cudaStream_t stream = 0
);