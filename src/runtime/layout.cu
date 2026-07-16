#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "layout.hpp"
#include <cstdio>

constexpr int HEADS = 32;
constexpr int HEAD_DIM = 64;
constexpr int BLOCK_SIZE = 256;

__global__
void flash_output_layout_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int tokens
){
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int total =
        tokens * HEADS * HEAD_DIM;

    if(idx >= total)
        return;

    // FlashAttention layout:
    //
    // [HEADS, TOKENS, HEAD_DIM]
    //
    // input index:
    // [head][token][dim]

    int dim =
        idx % HEAD_DIM;

    int token =
        (idx / HEAD_DIM) % tokens;

    int head =
        idx / (tokens * HEAD_DIM);

    // GEMM layout:
    //
    // [TOKENS, HEADS, HEAD_DIM]
    //
    // output index:
    // [token][head][dim]

    int output_idx =
        token * HEADS * HEAD_DIM
        +
        head * HEAD_DIM
        +
        dim;

    output[output_idx] =
        input[idx];
}

void launch_flash_output_layout(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int tokens
){
    int total =
        tokens * HEADS * HEAD_DIM;

    int blocks =
        (total + BLOCK_SIZE - 1)
        / BLOCK_SIZE;

    flash_output_layout_kernel<<<
        blocks,
        BLOCK_SIZE
    >>>(
        input,
        output,
        tokens
    );

    cudaError_t err =
        cudaGetLastError();

    if(err != cudaSuccess)
    {
        printf(
            "CUDA Error: %s\n",
            cudaGetErrorString(err)
        );

        exit(EXIT_FAILURE);
    }
}