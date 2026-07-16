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
        float* sin_table
    );

    void forward(
        const Tensor& input,
        Tensor& output
    ) const;

private:

    __nv_bfloat16* input_norm_;

    __nv_bfloat16* q_proj_;
    __nv_bfloat16* k_proj_;
    __nv_bfloat16* v_proj_;
    __nv_bfloat16* o_proj_;
    float* cos_table_;
    float* sin_table_;
};

} // namespace runtime