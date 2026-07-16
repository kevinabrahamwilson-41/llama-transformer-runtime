#ifndef ROPE_HPP
#define ROPE_HPP

#include <cuda_runtime.h>
#include <cuda_bf16.h>


// ======================================================
// Llama 3.2 1B RoPE Constants
// ======================================================
constexpr int ROPE_HEADS     = 32;
constexpr int ROPE_KV_HEADS  = 8;
constexpr int ROPE_HEAD_DIM  = 64;

constexpr int ROPE_BLOCK_SIZE = 256;

constexpr int ROPE_ROTARY_DIM = ROPE_HEAD_DIM / 2;


// ======================================================
// CUDA Error Checking
// ======================================================

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if(err != cudaSuccess)                                    \
    {                                                         \
        printf("CUDA Error %s:%d : %s\n",                     \
               __FILE__,                                     \
               __LINE__,                                     \
               cudaGetErrorString(err));                      \
        exit(EXIT_FAILURE);                                  \
    }                                                         \
} while(0)


// ======================================================
// RoPE Launcher
//
// Applies rotary positional embedding to:
//
// Q:
// [seq, heads, head_dim]
//
// K:
// [seq, kv_heads, head_dim]
//
// cos/sin:
// [seq, head_dim/2]
//
// ======================================================
void launch_rope_qkv(
    const __nv_bfloat16* q_in,
    __nv_bfloat16* q_out,

    const __nv_bfloat16* k_in,
    __nv_bfloat16* k_out,

    const __nv_bfloat16* v_in,
    __nv_bfloat16* v_out,

    float* cos_table,
    float* sin_table,

    int tokens
);


#endif