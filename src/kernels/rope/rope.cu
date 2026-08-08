#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "rope.hpp"
// ======================================================
// Llama 3.2 1B Constants
// ======================================================
constexpr int HEADS    = ROPE_HEADS;
constexpr int KV_HEADS = ROPE_KV_HEADS;
constexpr int HEAD_DIM = ROPE_HEAD_DIM;
constexpr int BLOCK_SIZE = ROPE_BLOCK_SIZE;
// RoPE dimensions
constexpr int ROTARY_DIM = ROPE_ROTARY_DIM;
// ======================================================
// RoPE rotation
//
// [x0,x1] -> [x0*cos - x1*sin,
//             x0*sin + x1*cos]
//
// ======================================================
__global__
void rope_q_kernel(
    const __nv_bfloat16* q_in,
    __nv_bfloat16* q_out,
    const float* cos_table,
    const float* sin_table,
    int tokens,
    int position_offset
){
    int warp_id =
        blockIdx.x * (blockDim.x / 32)
        + threadIdx.x / 32;

    int lane =
        threadIdx.x % 32;

    int total_warps =
        tokens * HEADS;

    if (warp_id >= total_warps)
        return;

    // One warp = one token position + one Q head
    int pos =
        warp_id / HEADS;

    int head =
        warp_id % HEADS;

    // One lane = one RoPE pair
    // pair 0  -> dimensions 0  and 32
    // pair 1  -> dimensions 1  and 33
    // ...
    // pair 31 -> dimensions 31 and 63
    int pair = lane;

    int input_offset =
        pos * HEADS * HEAD_DIM
        +
        head * HEAD_DIM;

    int output_offset =
        head * tokens * HEAD_DIM
        +
        pos * HEAD_DIM;

    int rope_pos =
        pos + position_offset;

    float c =
        cos_table[
            rope_pos * ROTARY_DIM + pair
        ];

    float s =
        sin_table[
            rope_pos * ROTARY_DIM + pair
        ];

    // Llama RoPE pairing:
    //
    // x0 = x[pair]
    // x1 = x[pair + 32]
    //
    // NOT:
    // x[pair*2], x[pair*2+1]

    float x0 =
        __bfloat162float(
            q_in[input_offset + pair]
        );

    float x1 =
        __bfloat162float(
            q_in[input_offset + pair + ROTARY_DIM]
        );

    // Rotary transformation
    float y0 =
        x0 * c - x1 * s;

    float y1 =
        x0 * s + x1 * c;

    q_out[
        output_offset + pair
    ] =
        __float2bfloat16(y0);

    q_out[
        output_offset + pair + ROTARY_DIM
    ] =
        __float2bfloat16(y1);
}
__global__
void rope_k_kernel(
    const __nv_bfloat16* k_in,
    __nv_bfloat16* k_out,
    const float* cos_table,
    const float* sin_table,
    int tokens,
    int position_offset
){
    int warp_id =
        blockIdx.x * (blockDim.x / 32)
        + threadIdx.x / 32;

    int lane =
        threadIdx.x % 32;

    int total_warps =
        tokens * KV_HEADS;

    if (warp_id >= total_warps)
        return;

    // One warp = one token position + one KV head
    int pos =
        warp_id / KV_HEADS;

    int head =
        warp_id % KV_HEADS;

    // One lane = one RoPE pair
    //
    // pair 0  -> dimensions 0  and 32
    // pair 1  -> dimensions 1  and 33
    // ...
    // pair 31 -> dimensions 31 and 63
    int pair = lane;

    int input_offset =
        pos * KV_HEADS * HEAD_DIM
        +
        head * HEAD_DIM;

    int output_offset =
        head * tokens * HEAD_DIM
        +
        pos * HEAD_DIM;

    int rope_pos =
        pos + position_offset;

    float c =
        cos_table[
            rope_pos * ROTARY_DIM + pair
        ];

    float s =
        sin_table[
            rope_pos * ROTARY_DIM + pair
        ];

    // Llama RoPE:
    //
    // x0 = x[pair]
    // x1 = x[pair + 32]
    //
    // NOT adjacent pairs (0,1), (2,3), etc.

    float x0 =
        __bfloat162float(
            k_in[input_offset + pair]
        );

    float x1 =
        __bfloat162float(
            k_in[input_offset + pair + ROTARY_DIM]
        );

    // Rotary transformation
    float y0 =
        x0 * c - x1 * s;

    float y1 =
        x0 * s + x1 * c;

    k_out[
        output_offset + pair
    ] =
        __float2bfloat16(y0);

    k_out[
        output_offset + pair + ROTARY_DIM
    ] =
        __float2bfloat16(y1);
}
__global__
void transpose_v_kernel(
    const __nv_bfloat16* v_in,
    __nv_bfloat16* v_out,
    int tokens
){
    int warp_id =
        blockIdx.x * (blockDim.x / 32)
        +
        threadIdx.x / 32;

    int lane =
        threadIdx.x % 32;

    int total_warps =
        tokens * KV_HEADS;

    if(warp_id >= total_warps)
        return;

    int pos =
        warp_id / KV_HEADS;

    int head =
        warp_id % KV_HEADS;

    int pair =
        lane;

    int input_offset =
        pos * KV_HEADS * HEAD_DIM
        +
        head * HEAD_DIM
        +
        pair * 2;

    int output_offset =
        head * tokens * HEAD_DIM
        +
        pos * HEAD_DIM
        +
        pair * 2;

    __nv_bfloat162 x =
        *reinterpret_cast<const __nv_bfloat162*>(
            &v_in[input_offset]
        );

    *reinterpret_cast<__nv_bfloat162*>(
        &v_out[output_offset]
    ) = x;
}

// ======================================================
// Launcher
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
    int tokens,
    int position_offset
){
    constexpr int WARPS_PER_BLOCK =
        BLOCK_SIZE / 32;

    // =====================================================
    // Q RoPE
    // =====================================================

    int q_warps =
        tokens * HEADS;

    int q_blocks =
        (q_warps + WARPS_PER_BLOCK - 1)
        / WARPS_PER_BLOCK;

    rope_q_kernel<<<
        q_blocks,
        BLOCK_SIZE
    >>>(
        q_in,
        q_out,
        cos_table,
        sin_table,
        tokens,
        position_offset
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // =====================================================
    // K RoPE
    // =====================================================

    int k_warps =
        tokens * KV_HEADS;

    int k_blocks =
        (k_warps + WARPS_PER_BLOCK - 1)
        / WARPS_PER_BLOCK;

    rope_k_kernel<<<
        k_blocks,
        BLOCK_SIZE
    >>>(
        k_in,
        k_out,
        cos_table,
        sin_table,
        tokens,
        position_offset
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

        // =====================================================
    // V LAYOUT CONVERSION
    // [tokens, 8, 64]
    //        ↓
    // [8, tokens, 64]
    // =====================================================

    int v_warps =
        tokens * KV_HEADS;

    int v_blocks =
        (v_warps + WARPS_PER_BLOCK - 1)
        / WARPS_PER_BLOCK;

    transpose_v_kernel<<<
        v_blocks,
        BLOCK_SIZE
    >>>(
        v_in,
        v_out,
        tokens
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
// ============================================================
// Helpers
// ============================================================

float bf16_to_float(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}

bool nearly_equal(float a, float b, float eps = 1e-2f)
{
    return fabsf(a - b) <= eps;
}