#pragma once
#include "tensor.hpp"
#include "attention.hpp"
#include "ffn.hpp"
namespace runtime {
class TransformerBlock {
public:
    TransformerBlock(
        Attention&& attention,
        FeedForward&& ffn
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
    Attention attention_;
    FeedForward ffn_;
};
} // namespace runtime