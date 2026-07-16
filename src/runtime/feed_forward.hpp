#pragma once

#include "tensor.hpp"

namespace runtime
{

class FeedForward
{
public:

    FeedForward(
        __nv_bfloat16* gate_proj,
        __nv_bfloat16* up_proj,
        __nv_bfloat16* down_proj
    );

    void forward(
        const Tensor& input,
        Tensor& output
    ) const;

private:

    __nv_bfloat16* gate_proj_;
    __nv_bfloat16* up_proj_;
    __nv_bfloat16* down_proj_;
};

} // namespace runtime