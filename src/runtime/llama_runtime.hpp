#pragma once
#include "tensor.hpp"
#include "transformer.hpp"
#include "weights.hpp"
#include <cuda_runtime.h>
namespace runtime{
class LlamaRuntime{
public:
    LlamaRuntime(
        const llama::LlamaWeights& weights,
        float* cos_table,
        float* sin_table
    );
    void forward(
        const Tensor& input,
        Tensor& logits
    ) const;
private:
    Transformer transformer_;
    Tensor final_norm_weight_;
    Tensor lm_head_weight_;
    int hidden_dim_;
    int vocab_size_;
};
}