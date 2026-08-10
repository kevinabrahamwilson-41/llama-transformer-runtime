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
    // ============================================================
    // PERFORMANCE STATISTICS
    // ============================================================
    struct PerformanceStats {
        // ============================================================
        // GENERATION
        // ============================================================
        int prompt_tokens = 0;
        int generated_tokens = 0;
        int total_tokens = 0;
        int initial_position = 0;
        int final_position = 0;
        int max_new_tokens = 0;
        std::string stop_reason;
        bool eos_encountered = false;
        bool eot_encountered = false;
        // ============================================================
        // KV CACHE
        // ============================================================
        int kv_cache_tokens = 0;
        int kv_cache_layers = 16;
        double kv_cache_mb = 0.0;
        // ============================================================
        // TOTAL TIMING
        // ============================================================
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
        double last_decode_ms = 0.0;
        double total_ms = 0.0;
        // ============================================================
        // THROUGHPUT
        // ============================================================
        double prefill_tokens_per_sec = 0.0;
        double decode_tokens_per_sec = 0.0;
        double overall_tokens_per_sec = 0.0;
        // ============================================================
        // PREFILL BREAKDOWN
        // ============================================================
        double prefill_embedding_ms = 0.0;
        double prefill_transformer_ms = 0.0;
        double prefill_final_rmsnorm_ms = 0.0;
        double prefill_lm_head_ms = 0.0;
        double prefill_sampling_ms = 0.0;
        double prefill_attention_ms = 0.0;
        double prefill_ffn_ms = 0.0;
        // ============================================================
        // DECODE BREAKDOWN
        // ============================================================
        double decode_embedding_ms = 0.0;
        double decode_transformer_ms = 0.0;
        double decode_final_rmsnorm_ms = 0.0;
        double decode_lm_head_ms = 0.0;
        double decode_sampling_ms = 0.0;
        double decode_attention_ms = 0.0;
        double decode_ffn_ms = 0.0;
        // ============================================================
        // STATIC MODEL MEMORY
        // ============================================================
        double model_weights_gb = 0.0;
        // ============================================================
        // RUNTIME MEMORY
        // ============================================================
        double activation_memory_mb = 0.0;
        double runtime_buffer_mb = 0.0;
        // ============================================================
        // COMPUTE ESTIMATES
        // ============================================================
        double estimated_flops = 0.0;
        double estimated_flops_per_token = 0.0;
        double estimated_gflops = 0.0;
        double estimated_tflops = 0.0;
        double arithmetic_intensity = 0.0;
    };
    double calculate_model_weights_gb() const;
    double calculate_kv_cache_mb(int tokens) const;
    double calculate_activation_memory_mb() const;
    double calculate_runtime_buffer_mb() const;
    LlamaRuntime(
        const std::string& weight_path,
        const std::string& tokenizer_path
    );
    ~LlamaRuntime();
    std::string generate(
        const std::string& prompt,
        int max_new_tokens = 128,
        bool use_topk = false,
        int k = 50
    );
    std::string chat(
        const std::string& user_message,
        int max_new_tokens = 128
    );
    void reset_conversation();
    const PerformanceStats& performance_stats() const{
        return performance_stats_;
    }
    void reset_performance_stats();
    void print_performance() const;
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
    PerformanceStats performance_stats_;
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
    Tensor final_norm_output_;
    Tensor last_hidden_state_;
    // -------------------------------
    // Configuration
    // -------------------------------
    int max_sequence_length_ = 4096;
    int conversation_position_ = 0;
    transformer::tokenizer::TokenSequence
    build_user_turn(
        const std::string& user_message
    );
    int forward_prefill(
        const transformer::tokenizer::TokenSequence& tokens,
        int start_position
    );
};
} // namespace runtime