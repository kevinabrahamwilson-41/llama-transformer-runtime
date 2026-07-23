// ======================================================
// Embedding Lookup Kernel (Vectorized BF16x2)
//
// Grid
// -----
// grid.x  = SEQ_LEN
//
// Block
// ------
// block.x = 256 threads
//
// One block copies one embedding vector
//
// Embedding:
// [VOCAB_SIZE][2048]
//
// Output:
// [SEQ_LEN][2048]
//
// Each thread copies BF16x2 values.
// ======================================================
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include "embedding.hpp"
__global__
void embedding_kernel(
    const int* __restrict__ tokens,
    const __nv_bfloat16* __restrict__ embedding_table,
    __nv_bfloat16* __restrict__ output
){
    //----------------------------------------------------
    // One block = one token
    //----------------------------------------------------
    const int token_pos = blockIdx.x;
    if (token_pos >= SEQ_LEN)
        return;
    //----------------------------------------------------
    // Vocabulary index
    //----------------------------------------------------
    const int token_id = tokens[token_pos];
    //----------------------------------------------------
    // Locate embedding row
    //----------------------------------------------------
    const __nv_bfloat16* __restrict__ src =
        embedding_table +
        token_id * HIDDEN_DIM;
    //----------------------------------------------------
    // Output row
    //----------------------------------------------------
    __nv_bfloat16* __restrict__ dst =
        output +
        token_pos * HIDDEN_DIM;
    //----------------------------------------------------
    // Vectorized BF16 copy
    // 2048 BF16
    // = 1024 BF16x2 values
    //----------------------------------------------------
    const __nv_bfloat162* __restrict__ src2 =
        reinterpret_cast<const __nv_bfloat162*>(src);
    __nv_bfloat162* __restrict__ dst2 =
        reinterpret_cast<__nv_bfloat162*>(dst);
#pragma unroll
    for (int i = threadIdx.x;
         i < HIDDEN_DIM / 2;
         i += BLOCK_SIZE){
        dst2[i] = src2[i];
    }
}
// ======================================================
// Launch embedding kernel
// ======================================================

void launch_embedding(
    const int* d_tokens,
    const __nv_bfloat16* d_embedding_table,
    __nv_bfloat16* d_output,
    cudaStream_t stream
)
{
    constexpr dim3 block(BLOCK_SIZE);
    const dim3 grid(SEQ_LEN);
    embedding_kernel<<<
        grid,
        block,
        0,
        stream
    >>>(
        d_tokens,
        d_embedding_table,
        d_output
    );
}