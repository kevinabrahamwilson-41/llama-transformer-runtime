#include "transformer.hpp"
#include <utility>
#include <cstdio>
namespace runtime
{
Transformer::Transformer(
    llama::LlamaWeights& weights,
    float* cos_table,
    float* sin_table,
    int max_seq_len
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
            sin_table,
            max_seq_len
        );
        FeedForward ffn(
            weights.layers[i].post_attention_layernorm,
            weights.layers[i].gate_proj,
            weights.layers[i].up_proj,
            weights.layers[i].down_proj
        );
        layers_.emplace_back(
            std::move(attention),
            std::move(ffn)
        );
    }
}

void Transformer::forward(
    const Tensor& input,
    Tensor& output,
    int position,
    int seq_len
)
{   printf("Transformer seq_len=%d\n", seq_len);
    const int tokens = seq_len;
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
            *next,
            position,
            seq_len
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
        current->numel() * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice
    );
}
}