#pragma once

#include "tensor.hpp"
#include "attention.hpp"
#include "ffn.hpp"

namespace runtime
{

class TransformerBlock
{
public:

    TransformerBlock(
        Attention attention,
        FeedForward ffn
    );

    void forward(
        const Tensor& input,
        Tensor& output
    ) const;

private:

    Attention attention_;
    FeedForward ffn_;
};

} // namespace runtime