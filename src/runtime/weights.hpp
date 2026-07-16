#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace llama
{

// ============================================================
// Llama 3.2 1B Instruct - FIXED MODEL CONSTANTS
// ============================================================

constexpr int NUM_LAYERS = 16;
constexpr int HIDDEN_SIZE = 2048;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int VOCAB_SIZE = 128256;

constexpr int NUM_ATTENTION_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 64;


// ============================================================
// Transformer Layer Weights
// ============================================================

struct TransformerLayerWeights
{
    __nv_bfloat16* input_layernorm;

    __nv_bfloat16* q_proj;
    __nv_bfloat16* k_proj;
    __nv_bfloat16* v_proj;
    __nv_bfloat16* o_proj;

    __nv_bfloat16* post_attention_layernorm;

    __nv_bfloat16* gate_proj;
    __nv_bfloat16* up_proj;
    __nv_bfloat16* down_proj;
};


// ============================================================
// Complete Llama Weights
// ============================================================

struct LlamaWeights
{
    // [VOCAB_SIZE, HIDDEN_SIZE]
    __nv_bfloat16* embed_tokens;

    TransformerLayerWeights layers[NUM_LAYERS];

    // [HIDDEN_SIZE]
    __nv_bfloat16* final_norm;

    // Llama uses tied embeddings.
    // lm_head == embed_tokens
    __nv_bfloat16* lm_head;
};


// ============================================================
// API
// ============================================================

void load_weights(
    const char* weights_path,
    LlamaWeights& weights
);

void free_weights(
    LlamaWeights& weights
);

} // namespace llama