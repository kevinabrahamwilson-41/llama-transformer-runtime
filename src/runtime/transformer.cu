#include "transformer.hpp"
#include <utility>
namespace runtime
{
Transformer::Transformer(
    llama::LlamaWeights& weights,
    float* cos_table,
    float* sin_table
){
    layers_.reserve(NUM_LAYERS);
    for(int i = 0; i < NUM_LAYERS; i++){
        Attention attention(
            weights.layers[i].input_layernorm,
            weights.layers[i].q_proj,
            weights.layers[i].k_proj,
            weights.layers[i].v_proj,
            weights.layers[i].o_proj,
            cos_table,
            sin_table
        );
        FeedForward ffn(
            weights.layers[i].post_attention_layernorm,
            weights.layers[i].gate_proj,
            weights.layers[i].up_proj,
            weights.layers[i].down_proj
        );
        layers_.emplace_back(
            attention,
            ffn
        );
    }
}

void Transformer::forward(
    const Tensor& input,
    Tensor& output
) const
{
    const int tokens =
        static_cast<int>(
            input.shape()[0]
        );
    Tensor buffer_a(
        {tokens, HIDDEN_SIZE},
        DataType::BF16
    );
    Tensor buffer_b(
        {tokens, HIDDEN_SIZE},
        DataType::BF16
    );
    const Tensor* current =
        &input;
    Tensor* next =
        &buffer_a;
    for(int i = 0; i < NUM_LAYERS; i++){
        layers_[i].forward(
            *current,
            *next
        );
        current = next;
        if(next == &buffer_a)
        {
            next = &buffer_b;
        }
        else
        {
            next = &buffer_a;
        }
    }
    cudaMemcpy(
        output.data_bf16(),
        current->data_bf16(),
        input.numel() * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice
    );
}
}