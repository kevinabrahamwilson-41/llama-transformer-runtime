#include "transformer.hpp"
#include <utility>
#include <cstdio>
#include "debug_dump.hpp"
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
            max_seq_len,
            i
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

// ============================================================
// KV CACHE GETTERS
// ============================================================
const __nv_bfloat16* Transformer::get_key_cache(int layer_idx) const{
    return layers_[layer_idx].get_key_cache();
}
const __nv_bfloat16* Transformer::get_value_cache(int layer_idx) const{
    return layers_[layer_idx].get_value_cache();
}
void Transformer::forward(
    const Tensor& input,
    Tensor& output,
    int position,
    int seq_len
){  
    printf("Transformer seq_len=%d\n", seq_len);
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
    if (position == 0) {
        save_checkpoint(
            "transformer_entry",
            current->data_bf16(),
            tokens * HIDDEN_SIZE
        );
    }
    Tensor* next =
        &buffer_a;
    for(int i = 0; i < NUM_LAYERS; i++){
        if(i == 0 && position == 0){
            save_checkpoint(
                "before_layer0",
                current->data_bf16(),
                tokens * HIDDEN_SIZE
            );
        }
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