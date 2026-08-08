#pragma once

#include "tensor.hpp"
#include <cuda_bf16.h>

namespace runtime
{

class Attention
{
public:

    Attention(
        __nv_bfloat16* input_norm,
        __nv_bfloat16* q_proj,
        __nv_bfloat16* k_proj,
        __nv_bfloat16* v_proj,
        __nv_bfloat16* o_proj,
        float* cos_table,
        float* sin_table,
        int max_seq_len,
        int layer_idx
    );
    void forward(
        const Tensor& input,
        Tensor& output,
        int position,
        int seq_len
    );
    const __nv_bfloat16* get_key_cache() const;
    const __nv_bfloat16* get_value_cache() const;

private:

    __nv_bfloat16* input_norm_;

    __nv_bfloat16* q_proj_;
    __nv_bfloat16* k_proj_;
    __nv_bfloat16* v_proj_;
    __nv_bfloat16* o_proj_;
    float* cos_table_;
    float* sin_table_;
    // ============================
    // KV Cache
    //
    // K: [8, max_seq_len, 64]
    // V: [8, max_seq_len, 64]
    // ============================
    Tensor key_cache_;
    Tensor value_cache_;
    int max_seq_len_;
    int layer_idx_;
};

} // namespace runtime