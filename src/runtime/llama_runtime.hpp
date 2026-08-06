#pragma once
#include <memory>
#include <string>
#include <vector>
#include "transformer.hpp"
#include "tensor.hpp"
#include "weights.hpp"
#include "../../models/tokenizer/tokenizer.hpp"
namespace runtime
{
class LlamaRuntime
{
public:
    LlamaRuntime(
        const std::string& weight_path,
        const std::string& tokenizer_path
    );
    LlamaRuntime();
    std::string generate(
        const std::string& prompt,
        int max_new_tokens = 128,
        bool use_topk = false,
        int k = 50
    );
private:
    void initialize();
    void build_rope_tables();
    int decode_forward(
        int token_id,
        int position
    );
private:
    // -------------------------------
    // Tokenizer
    // -------------------------------
    std::unique_ptr<transformer::tokenizer::Tokenizer> tokenizer_;
    // -------------------------------
    // Model weights
    // -------------------------------
    llama::LlamaWeights weights_;
    // -------------------------------
    // Transformer
    // -------------------------------
    std::unique_ptr<Transformer> transformer_;
    // -------------------------------
    // RoPE lookup tables
    // -------------------------------
    float* cos_table_ = nullptr;
    float* sin_table_ = nullptr;
    // -------------------------------
    // Runtime tensors
    // -------------------------------
    Tensor hidden_states_;
    Tensor logits_;
    Tensor transformer_output_;
    int* token_buffer_ = nullptr;
    // -------------------------------
    // Configuration
    // -------------------------------
    int max_sequence_length_ = 4096;
        int forward_prefill(
        const transformer::tokenizer::TokenSequence& tokens
    );
};
} // namespace runtime