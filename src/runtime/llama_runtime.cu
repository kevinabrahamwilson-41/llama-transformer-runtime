#include "llama_runtime.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <string>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include "../kernels/embedding/embedding.hpp"
#include "../kernels/rmsnorm/rmsnorm.hpp"
#include "../kernels/gemm/gemm_2048x128256.hpp"
#include "../kernels/sampling/argmax_128256.hpp"
#include "attention.hpp"
#include "ffn.hpp"
#include <iostream>
namespace fs = std::filesystem;
namespace runtime{
    double LlamaRuntime::calculate_model_weights_gb() const{
        const std::filesystem::path weights_path =
            "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama_weights.bin";
        if (!std::filesystem::exists(weights_path)) {
            return 0.0;
        }
        const std::uintmax_t file_size =
            std::filesystem::file_size(weights_path);
        return static_cast<double>(file_size) / 1e9;
    }
    std::vector<float> LlamaRuntime::generate_logits(
    const std::string& prompt
    ) {
        transformer::tokenizer::TokenSequence tokens =
            tokenizer_->encode(prompt);
        if (tokens.empty()) {
            throw std::runtime_error(
                "generate_logits: prompt produced no tokens"
            );
        }
        reset_performance_stats();
        forward_prefill(tokens, 0);
        const int64_t num_logits = logits_.numel();
        std::vector<__nv_bfloat16> host_logits(
            static_cast<size_t>(num_logits)
        );
        cudaError_t error = cudaMemcpy(
            host_logits.data(),
            logits_.data_bf16(),
            logits_.bytes(),
            cudaMemcpyDeviceToHost
        );
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("generate_logits cudaMemcpy failed: ") +
                cudaGetErrorString(error)
            );
        }
        std::vector<float> result(
            static_cast<size_t>(num_logits)
        );
        for (int64_t i = 0; i < num_logits; ++i) {
            result[static_cast<size_t>(i)] =
                __bfloat162float(host_logits[static_cast<size_t>(i)]);
        }
        return result;
    }
    std::vector<int> LlamaRuntime::generate_token_ids(
        const std::string& prompt,
        int max_new_tokens
    ) {
        if(max_new_tokens <= 0){
            return {};
        }
        transformer::tokenizer::TokenSequence tokens =
            tokenizer_->encode(prompt);
        if(tokens.empty()){
            throw std::runtime_error(
                "generate_token_ids: prompt produced no tokens"
            );
        }
        reset_performance_stats();
        int position = 0;
        int next_token =
            forward_prefill(
                tokens,
                position
            );
        // DEBUG: first token predicted from prefill
        std::cout << "Prefill next token: "
                << next_token
                << "\n";
        position +=
            static_cast<int>(tokens.size());
        std::vector<int> generated_tokens;
        generated_tokens.reserve(
            static_cast<size_t>(max_new_tokens)
        );
        for(int i = 0; i < max_new_tokens; i++){
            if(next_token == tokenizer_->eot_id()){
                break;
            }
            if(next_token == tokenizer_->eos_id()){
                break;
            }
            generated_tokens.push_back(next_token);
            int token =
                decode_forward(
                    next_token,
                    position
                );
            // DEBUG: result of first decode step only
            if(i == 0){
                std::cout << "First decode token: "
                        << token
                        << "\n";
            }
            position++;
            next_token = token;
        }
        return generated_tokens;
    }
    double LlamaRuntime::calculate_runtime_buffer_mb() const{
        constexpr size_t ROTARY_DIM = 32;
        size_t bytes = 0;
        // =========================================================
        // Token buffer
        // =========================================================
        if(token_buffer_ != nullptr){
            bytes += sizeof(int);
        }
        // =========================================================
        // RoPE tables
        // =========================================================
        if(cos_table_ != nullptr){
            bytes +=
                static_cast<size_t>(max_sequence_length_)
                * ROTARY_DIM
                * sizeof(float);
        }
        if(sin_table_ != nullptr){
            bytes +=
                static_cast<size_t>(max_sequence_length_)
                * ROTARY_DIM
                * sizeof(float);
        }
        return static_cast<double>(bytes)
            / (1024.0 * 1024.0);
    }
    double LlamaRuntime::calculate_kv_cache_mb(int tokens) const{
        constexpr double LAYERS = 16.0;
        constexpr double KV_HEADS = 8.0;
        constexpr double HEAD_DIM = 64.0;
        constexpr double KV = 2.0;
        const double bytes =
            static_cast<double>(tokens)
            * LAYERS
            * KV_HEADS
            * HEAD_DIM
            * KV
            * sizeof(nv_bfloat16);
        return bytes / (1024.0 * 1024.0);
    }
    double LlamaRuntime::calculate_activation_memory_mb() const{
        size_t total_bytes = 0;
        total_bytes += hidden_states_.bytes();
        total_bytes += transformer_output_.bytes();
        total_bytes += final_norm_output_.bytes();
        total_bytes += last_hidden_state_.bytes();
        total_bytes += logits_.bytes();
        return static_cast<double>(total_bytes)
            / (1024.0 * 1024.0);
    }
    class CudaTimer{
    public:
        CudaTimer(){
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
        }
        ~CudaTimer(){
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }
        void start(){
            cudaEventRecord(start_);
        }
        double stop_ms(){
            cudaEventRecord(stop_);
            cudaEventSynchronize(stop_);
            float ms = 0.0f;
            cudaEventElapsedTime(
                &ms,
                start_,
                stop_
            );
            return static_cast<double>(ms);
        }
    private:
        cudaEvent_t start_{};
        cudaEvent_t stop_{};
    };
    void LlamaRuntime::print_performance() const{
        const PerformanceStats& s = performance_stats_;
        printf("\n");
        printf("============================================================\n");
        printf("              LLAMA RUNTIME PERFORMANCE\n");
        printf("============================================================\n");
        // ========================================================
        // Generation
        // ========================================================
        printf("\n[Generation]\n");
        printf("  Prompt tokens        : %d\n",
            s.prompt_tokens);
        printf("  Generated tokens     : %d\n",
            s.generated_tokens);
        printf("  Total tokens         : %d\n",
            s.total_tokens);
        printf("  Max new tokens       : %d\n",
            s.max_new_tokens);
        printf("  Initial position     : %d\n",
            s.initial_position);
        printf("  Final position       : %d\n",
            s.final_position);
        printf("  Stop reason          : %s\n",
            s.stop_reason.empty()
                ? "MAX_TOKENS"
                : s.stop_reason.c_str());
        printf("  EOS encountered      : %s\n",
            s.eos_encountered ? "yes" : "no");
        printf("  EOT encountered      : %s\n",
            s.eot_encountered ? "yes" : "no");
        // ========================================================
        // KV Cache
        // ========================================================
        printf("\n[KV Cache]\n");
        printf("  KV cache tokens      : %d\n",
            s.kv_cache_tokens);
        printf("  KV cache layers      : %d\n",
            s.kv_cache_layers);
        printf("  KV cache memory      : %.3f MB\n",
            s.kv_cache_mb);
        // ========================================================
        // Timing
        // ========================================================
        printf("\n[Timing]\n");
        printf("  Prefill              : %.3f ms\n",
            s.prefill_ms);
        printf("  Decode               : %.3f ms\n",
            s.decode_ms);
        printf("  Last decode token    : %.3f ms\n",
            s.last_decode_ms);
        printf("  Total                : %.3f ms\n",
            s.total_ms);
        // ========================================================
        // Prefill Breakdown
        // ========================================================
        printf("\n[Prefill Breakdown]\n");
        printf("  Embedding            : %.3f ms\n",
            s.prefill_embedding_ms);
        printf("  Transformer          : %.3f ms\n",
            s.prefill_transformer_ms);
        printf("  Attention            : %.3f ms\n",
            s.prefill_attention_ms);
        printf("  FFN                  : %.3f ms\n",
            s.prefill_ffn_ms);
        printf("  Final RMSNorm        : %.3f ms\n",
            s.prefill_final_rmsnorm_ms);
        printf("  LM Head              : %.3f ms\n",
            s.prefill_lm_head_ms);
        printf("  Sampling             : %.3f ms\n",
            s.prefill_sampling_ms);
        // ========================================================
        // Decode Breakdown
        // ========================================================
        printf("\n[Decode Breakdown]\n");
        printf("  Embedding            : %.3f ms\n",
            s.decode_embedding_ms);
        printf("  Transformer          : %.3f ms\n",
            s.decode_transformer_ms);
        printf("  Attention            : %.3f ms\n",
            s.decode_attention_ms);
        printf("  FFN                  : %.3f ms\n",
            s.decode_ffn_ms);
        printf("  Final RMSNorm        : %.3f ms\n",
            s.decode_final_rmsnorm_ms);
        printf("  LM Head              : %.3f ms\n",
            s.decode_lm_head_ms);
        printf("  Sampling             : %.3f ms\n",
            s.decode_sampling_ms);
        // ========================================================
        // Throughput
        // ========================================================
        printf("\n[Throughput]\n");
        printf("  Prefill              : %.3f tokens/s\n",
            s.prefill_tokens_per_sec);
        printf("  Decode               : %.3f tokens/s\n",
            s.decode_tokens_per_sec);
        printf("  Overall              : %.3f tokens/s\n",
            s.overall_tokens_per_sec);
        // ========================================================
        // Memory
        // ========================================================
        printf("\n[Memory]\n");
        printf("  Model weights        : %.3f GB\n",
            s.model_weights_gb);
        printf("  KV cache             : %.3f MB\n",
            s.kv_cache_mb);
        printf("  Activations          : %.3f MB\n",
            s.activation_memory_mb);
        printf("  Runtime buffers      : %.3f MB\n",
            s.runtime_buffer_mb);
        // ========================================================
        // End
        // ========================================================
        printf("\n");
        printf("============================================================\n");
    }
    void LlamaRuntime::build_rope_tables(){
        constexpr int ROTARY_DIM = 32;
        constexpr int HEAD_DIM = 64;
        constexpr float ROPE_THETA = 500000.0f;
        constexpr float FACTOR = 32.0f;
        constexpr float LOW_FREQ_FACTOR = 1.0f;
        constexpr float HIGH_FREQ_FACTOR = 4.0f;
        constexpr float ORIGINAL_CONTEXT_LENGTH = 8192.0f;
        const int seq_len = max_sequence_length_;
        size_t table_size =
            seq_len * ROTARY_DIM;
        // ==========================================
        // Allocate GPU RoPE tables
        // ==========================================
        cudaError_t err;
        err = cudaMalloc(
            &cos_table_,
            table_size * sizeof(float)
        );
        if(err != cudaSuccess){
            throw std::runtime_error(
                "Failed allocating RoPE cos table"
            );
        }
        err = cudaMalloc(
            &sin_table_,
            table_size * sizeof(float)
        );
        if(err != cudaSuccess){
            throw std::runtime_error(
                "Failed allocating RoPE sin table"
            );
        }
        // ==========================================
        // Create host tables
        // ==========================================
        std::vector<float> host_cos(
            table_size
        );
        std::vector<float> host_sin(
            table_size
        );
        // ==========================================
        // Llama 3.2 RoPE frequency calculation
        // ==========================================
        for(int pair = 0; pair < ROTARY_DIM; pair++){
            float exponent =
                static_cast<float>(2 * pair)/static_cast<float>(HEAD_DIM);
            float inv_freq =
                1.0f /
                std::pow(
                    static_cast<float>(ROPE_THETA),
                    exponent
                );
            float wavelen =
                2.0f *
                static_cast<float>(M_PI)
                /
                inv_freq;
            float low_freq_wavelen =
                ORIGINAL_CONTEXT_LENGTH /
                LOW_FREQ_FACTOR;
            float high_freq_wavelen =
                ORIGINAL_CONTEXT_LENGTH /
                HIGH_FREQ_FACTOR;
            float inv_freq_llama;
            if(wavelen < high_freq_wavelen){
                // high frequency
                inv_freq_llama =
                    inv_freq;
            }
            else if(wavelen <= low_freq_wavelen){
                // smooth interpolation
                float smooth_factor =
                    (
                        ORIGINAL_CONTEXT_LENGTH / wavelen
                        -
                        LOW_FREQ_FACTOR
                    )
                    /
                    (
                        HIGH_FREQ_FACTOR
                        -
                        LOW_FREQ_FACTOR
                    );
                inv_freq_llama =
                    (1.0f - smooth_factor)
                    *
                    (inv_freq / FACTOR)
                    +
                    smooth_factor * inv_freq;
            }
            else{
                // low frequency
                inv_freq_llama =
                    inv_freq / FACTOR;
            }
            for(int pos = 0; pos < seq_len; pos++){
                float angle =
                    static_cast<float>(pos)
                    *
                    inv_freq_llama;
                host_cos[
                    pos * ROTARY_DIM + pair
                ] =
                    std::cos(angle);
                host_sin[
                    pos * ROTARY_DIM + pair
                ] =
                    std::sin(angle);
            }
        }
        // ==========================================
        // Copy tables to GPU
        // ==========================================
        cudaMemcpy(
            cos_table_,
            host_cos.data(),
            table_size * sizeof(float),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            sin_table_,
            host_sin.data(),
            table_size * sizeof(float),
            cudaMemcpyHostToDevice
        );
        printf(
            "[Runtime] RoPE tables created (%d tokens)\n",
            seq_len
        );
    }
    transformer::tokenizer::TokenSequence
    LlamaRuntime::build_user_turn(
        const std::string& user_message
    ){
        using transformer::tokenizer::TokenSequence;
        using transformer::tokenizer::EncodeOptions;
        TokenSequence tokens;
        // ==================================================
        // Encoding text inside the chat template
        //
        // Do NOT automatically add BOS/EOS here.
        // The chat template adds special tokens explicitly.
        // ==================================================
        EncodeOptions no_specials;
        no_specials.bos = false;
        no_specials.eos = false;
        // ==================================================
        // 1. BOS — only at beginning of conversation
        // ==================================================
        if(conversation_position_ == 0){
            tokens.push_back(
                tokenizer_->bos_id()
            );
        }
        // ==================================================
        // 2. USER HEADER
        //
        // <|start_header_id|>
        // user
        // <|end_header_id|>
        // \n
        // ==================================================

        tokens.push_back(
            tokenizer_->start_header_id()
        );
        auto user_tokens =
            tokenizer_->encode(
                "user",
                no_specials
            );
        tokens.insert(
            tokens.end(),
            user_tokens.begin(),
            user_tokens.end()
        );
        tokens.push_back(
            tokenizer_->end_header_id()
        );
        // Header newline
        auto newline_tokens =
            tokenizer_->encode(
                "\n",
                no_specials
            );
        tokens.insert(
            tokens.end(),
            newline_tokens.begin(),
            newline_tokens.end()
        );
        // ==================================================
        // 3. USER MESSAGE
        // ==================================================
        auto message_tokens =
            tokenizer_->encode(
                user_message,
                no_specials
        );
        tokens.insert(
            tokens.end(),
            message_tokens.begin(),
            message_tokens.end()
        );
        // ==================================================
        // 4. END OF USER TURN
        // ==================================================
        tokens.push_back(
            tokenizer_->eot_id()
        );
        // ==================================================
        // 5. ASSISTANT HEADER
        //
        // <|start_header_id|>
        // assistant
        // <|end_header_id|>
        // \n
        // ==================================================
        tokens.push_back(
            tokenizer_->start_header_id()
        );
        auto assistant_tokens =
            tokenizer_->encode(
                "assistant",
                no_specials
        );
        tokens.insert(
            tokens.end(),
            assistant_tokens.begin(),
            assistant_tokens.end()
        );
        tokens.push_back(
            tokenizer_->end_header_id()
        );
        // Header newline
        tokens.insert(
            tokens.end(),
            newline_tokens.begin(),
            newline_tokens.end()
        );
        return tokens;
    }
    std::string LlamaRuntime::chat(
        const std::string& user_message,
        int max_new_tokens
    ){
        // ==================================================
        // 0. RESET PERFORMANCE STATISTICS
        // ==================================================
        reset_performance_stats();
        PerformanceStats& stats =
            performance_stats_;
        stats.initial_position =
            conversation_position_;
        stats.max_new_tokens =
            max_new_tokens;
        // ==================================================
        // 1. BUILD USER TURN
        // ==================================================
        transformer::tokenizer::TokenSequence tokens =
            build_user_turn(user_message);
        stats.prompt_tokens =
            static_cast<int>(tokens.size());
        // ==================================================
        // 2. CHECK CONTEXT LENGTH
        // ==================================================
        if(
            conversation_position_
            + static_cast<int>(tokens.size())
            + max_new_tokens
            >= max_sequence_length_
        ){
            throw std::runtime_error(
                "Conversation exceeds maximum sequence length"
            );
        }
        // ==================================================
        // 3. PREFILL
        // ==================================================
        int next_token =
            forward_prefill(
                tokens,
                conversation_position_
            );
        conversation_position_ +=
            static_cast<int>(tokens.size());
        // ==================================================
        // 4. AUTOREGRESSIVE DECODE
        // ==================================================
        transformer::tokenizer::TokenSequence
            response_tokens;
        double decode_total_ms = 0.0;
        for(int i = 0; i < max_new_tokens; i++){
            int token =
                decode_forward(
                    next_token,
                    conversation_position_
                );
            // decode_forward() updates the timing
            // for ONE generated token.
            decode_total_ms +=
                performance_stats_.last_decode_ms;
            // --------------------------------------------------
            // EOT
            // --------------------------------------------------
            if(token == tokenizer_->eot_id()){
                stats.eot_encountered = true;
                stats.stop_reason = "EOT";
                conversation_position_++;
                break;
            }
            // --------------------------------------------------
            // EOS
            // --------------------------------------------------
            if(token == tokenizer_->eos_id()){
                stats.eos_encountered = true;
                stats.stop_reason = "EOS";
                conversation_position_++;
                break;
            }
            // --------------------------------------------------
            // Store generated token
            // --------------------------------------------------
            response_tokens.push_back(token);
            stats.generated_tokens++;
            conversation_position_++;
            next_token = token;
        }
        // ==================================================
        // 5. FINAL STATISTICS
        // ==================================================
        stats.total_tokens =
            stats.prompt_tokens
            +
            stats.generated_tokens;
        stats.final_position =
            conversation_position_;
        stats.kv_cache_tokens =
            conversation_position_;
        stats.kv_cache_mb =
            calculate_kv_cache_mb(
                stats.kv_cache_tokens
        );
        stats.decode_ms =
            decode_total_ms;
        stats.total_ms = stats.prefill_ms + stats.decode_ms;
        stats.model_weights_gb = calculate_model_weights_gb();
        stats.activation_memory_mb = calculate_activation_memory_mb();
        stats.runtime_buffer_mb = calculate_runtime_buffer_mb();
        stats.kv_cache_mb = calculate_kv_cache_mb(stats.kv_cache_tokens);
        // --------------------------------------------------
        // Throughput
        // --------------------------------------------------
        if(stats.prefill_ms > 0.0){
            stats.prefill_tokens_per_sec =
                static_cast<double>(stats.prompt_tokens)
                /
                (stats.prefill_ms / 1000.0);
        }
        if(stats.decode_ms > 0.0){
            stats.decode_tokens_per_sec =
                static_cast<double>(stats.generated_tokens)
                /
                (stats.decode_ms / 1000.0);
        }
        if(stats.total_ms > 0.0){
            stats.overall_tokens_per_sec =
                static_cast<double>(stats.total_tokens)
                /
                (stats.total_ms / 1000.0);
        }
        // ==================================================
        // 6. DECODE ONLY ASSISTANT RESPONSE
        // ==================================================
        std::string response =
            tokenizer_->decode(response_tokens);
        // ==================================================
        // 7. PRINT PERFORMANCE REPORT
        // ==================================================
        print_performance();
        return response;
    }
    void LlamaRuntime::reset_conversation(){
        conversation_position_ = 0;
    }
    void LlamaRuntime::reset_performance_stats(){
        performance_stats_ = PerformanceStats{};
    }
    std::string LlamaRuntime::generate(
        const std::string& prompt,
        int max_new_tokens,
        bool use_topk,
        int k
    ){
        // ==================================================
        // 0. RESET PERFORMANCE STATISTICS
        // ==================================================
        reset_performance_stats();
        PerformanceStats& stats =
            performance_stats_;
        stats.initial_position = 0;
        stats.max_new_tokens = max_new_tokens;
        // ==================================================
        // 1. ENCODE PROMPT
        // ==================================================
        transformer::tokenizer::TokenSequence tokens =
            tokenizer_->encode(prompt);
        stats.prompt_tokens =
            static_cast<int>(tokens.size());
        transformer::tokenizer::TokenSequence generated_tokens;
        // Copy prompt tokens
        for(auto token : tokens){
            generated_tokens.push_back(token);
        }
        // ==================================================
        // 2. PREFILL
        // ==================================================
        int next_token =
            forward_prefill(tokens, 0);
        // All prompt tokens are now in the KV cache.
        int position =
            static_cast<int>(tokens.size());
        // ==================================================
        // 3. AUTOREGRESSIVE DECODE
        // ==================================================
        double decode_total_ms = 0.0;
        for(int i = 0; i < max_new_tokens; i++){
            int token =
                decode_forward(
                    next_token,
                    position
                );
            // decode_forward() records timing
            // for this generated token.
            decode_total_ms +=
                performance_stats_.last_decode_ms;
            // --------------------------------------------------
            // EOS
            // --------------------------------------------------
            if(token == tokenizer_->eos_id()){
                stats.eos_encountered = true;
                stats.stop_reason = "EOS";
                position++;
                break;
            }
            // --------------------------------------------------
            // Store generated token
            // --------------------------------------------------
            generated_tokens.push_back(token);
            stats.generated_tokens++;
            next_token = token;
            position++;
        }
        // ==================================================
        // 4. FINAL PERFORMANCE STATISTICS
        // ==================================================
        stats.total_tokens =
            stats.prompt_tokens
            +
            stats.generated_tokens;
        stats.final_position =
            position;
        stats.kv_cache_tokens =
            position;
        stats.decode_ms =
            decode_total_ms;
        stats.total_ms =
            stats.prefill_ms
            +
            stats.decode_ms;
        // --------------------------------------------------
        // Throughput
        // --------------------------------------------------
        if(stats.prefill_ms > 0.0){
            stats.prefill_tokens_per_sec =
                static_cast<double>(stats.prompt_tokens)
                /
                (stats.prefill_ms / 1000.0);
        }
        if(stats.decode_ms > 0.0){
            stats.decode_tokens_per_sec =
                static_cast<double>(stats.generated_tokens)
                /
                (stats.decode_ms / 1000.0);
        }
        if(stats.total_ms > 0.0){
            stats.overall_tokens_per_sec =
                static_cast<double>(stats.total_tokens)
                /
                (stats.total_ms / 1000.0);
        }
        // ==================================================
        // 5. DECODE OUTPUT
        // ==================================================
        std::string response =
            tokenizer_->decode(generated_tokens);
        // ==================================================
        // 6. PRINT PERFORMANCE
        // ==================================================
        print_performance();
        return response;
    }
    int LlamaRuntime::decode_forward(
        int token_id,
        int position
    ){
        // ==================================================
        // 1. TOTAL DECODE TOKEN TIMER
        // ==================================================
        CudaTimer total_timer;
        total_timer.start();
        // ==================================================
        // 2. Copy token ID to GPU
        // ==================================================
        cudaMemcpy(
            token_buffer_,
            &token_id,
            sizeof(int),
            cudaMemcpyHostToDevice
        );
        // ==================================================
        // 3. TOKEN EMBEDDING
        // ==================================================
        CudaTimer timer_embedding;
        timer_embedding.start();
        launch_embedding(
            token_buffer_,
            weights_.embed_tokens,
            hidden_states_.data_bf16(),
            1
        );
        cudaDeviceSynchronize();
        performance_stats_.decode_embedding_ms +=
            timer_embedding.stop_ms();
        // ==================================================
        // 4. TRANSFORMER
        // ==================================================
        CudaTimer timer_transformer;
        timer_transformer.start();
        // reset per-layer timers
        Attention::reset_attention_timing();
        FeedForward::reset_ffn_timing();
        transformer_->forward(
            hidden_states_,
            transformer_output_,
            position,
            1
        );
        cudaDeviceSynchronize();
        performance_stats_.decode_transformer_ms +=
            timer_transformer.stop_ms();
        // accumulate per-layer timings recorded during transformer forward
        performance_stats_.decode_attention_ms +=
            Attention::get_accumulated_attention_ms();
        performance_stats_.decode_ffn_ms +=
            FeedForward::get_accumulated_ffn_ms();
        // ==================================================
        // 5. FINAL RMSNorm
        // ==================================================
        CudaTimer timer_rmsnorm;
        timer_rmsnorm.start();
        rmsnorm_launch<llama::HIDDEN_SIZE>(
            transformer_output_.data_bf16(),
            weights_.final_norm,
            final_norm_output_.data_bf16(),
            1,
            1e-5f
        );
        cudaDeviceSynchronize();
        performance_stats_.decode_final_rmsnorm_ms +=
            timer_rmsnorm.stop_ms();
        // ==================================================
        // 6. LM HEAD
        // ==================================================
        CudaTimer timer_lm_head;
        timer_lm_head.start();
        launch_gemm_2048x128256(
            final_norm_output_.data_bf16(),
            weights_.lm_head,
            logits_.data_bf16(),
            1,
            llama::VOCAB_SIZE,
            llama::HIDDEN_SIZE
        );
        cudaDeviceSynchronize();
        performance_stats_.decode_lm_head_ms +=
            timer_lm_head.stop_ms();
        // ==================================================
        // 7. ARGMAX / SAMPLING
        // ==================================================
        int next_token = 0;
        int* d_next_token = nullptr;
        cudaMalloc(
            &d_next_token,
            sizeof(int)
        );
        CudaTimer timer_sampling;
        timer_sampling.start();
        launch_argmax(
            logits_.data_bf16(),
            d_next_token,
            0
        );
        cudaMemcpy(
            &next_token,
            d_next_token,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
        cudaFree(d_next_token);
        cudaDeviceSynchronize();
        performance_stats_.decode_sampling_ms +=
            timer_sampling.stop_ms();
        // ==================================================
        // 8. TOTAL TIME FOR THIS DECODE TOKEN
        // ==================================================
        performance_stats_.last_decode_ms =
            total_timer.stop_ms();
        return next_token;
    }

    int LlamaRuntime::forward_prefill(
        const transformer::tokenizer::TokenSequence& tokens,
        int start_position
    ){
        int seq_len = static_cast<int>(tokens.size());
        // ==================================================
        // 1. TOTAL PREFILL TIMER
        // ==================================================
        CudaTimer total_timer;
        total_timer.start();
        // ==================================================
        // 2. Copy prompt tokens to GPU
        // ==================================================
        int* prompt_tokens_gpu = nullptr;
        cudaMalloc(
            &prompt_tokens_gpu,
            seq_len * sizeof(int)
        );
        cudaMemcpy(
            prompt_tokens_gpu,
            tokens.data(),
            seq_len * sizeof(int),
            cudaMemcpyHostToDevice
        );
        // ==================================================
        // 3. TOKEN EMBEDDING
        //
        // tokens
        //    |
        //    v
        // hidden_states [seq_len, 2048]
        // ==================================================
        CudaTimer timer_embedding;
        timer_embedding.start();
        launch_embedding(
            prompt_tokens_gpu,
            weights_.embed_tokens,
            hidden_states_.data_bf16(),
            seq_len
        );
        cudaDeviceSynchronize();
        performance_stats_.prefill_embedding_ms =
            timer_embedding.stop_ms();
        // ==================================================
        // 4. TRANSFORMER PREFILL
        //
        // This fills the KV cache.
        //
        // Q = [32, seq_len, 64]
        // K = [8,  seq_len, 64]
        // V = [8,  seq_len, 64]
        //
        // Cache positions:
        // start_position ... start_position + seq_len - 1
        // ==================================================
        CudaTimer timer_transformer;
        timer_transformer.start();
        // reset per-layer timers
        Attention::reset_attention_timing();
        FeedForward::reset_ffn_timing();
        transformer_->forward(
            hidden_states_,
            transformer_output_,
            start_position,
            seq_len
        );
        cudaDeviceSynchronize();
        performance_stats_.prefill_transformer_ms =
            timer_transformer.stop_ms();
        // record per-layer timings collected during transformer forward
        performance_stats_.prefill_attention_ms =
            Attention::get_accumulated_attention_ms();
        performance_stats_.prefill_ffn_ms =
            FeedForward::get_accumulated_ffn_ms();
        // ==================================================
        // 5. Extract LAST TOKEN hidden state
        //
        // transformer_output:
        // [seq_len, 2048]
        //
        // We only need:
        // [last token, 2048]
        // ==================================================
        cudaMemcpy(
            last_hidden_state_.data_bf16(),
            transformer_output_.data_bf16()
                + (seq_len - 1) * llama::HIDDEN_SIZE,
            llama::HIDDEN_SIZE * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice
        );
        cudaDeviceSynchronize();
        // ==================================================
        // 6. FINAL RMSNorm
        // ==================================================
        CudaTimer timer_rmsnorm;
        timer_rmsnorm.start();
        rmsnorm_launch<llama::HIDDEN_SIZE>(
            last_hidden_state_.data_bf16(),
            weights_.final_norm,
            final_norm_output_.data_bf16(),
            1,
            1e-5f
        );
        cudaDeviceSynchronize();
        performance_stats_.prefill_final_rmsnorm_ms =
            timer_rmsnorm.stop_ms();
        // ==================================================
        // 7. LM HEAD
        //
        // [1, 2048]
        //      x
        // [2048, 128256]
        //      |
        //      v
        // [1, 128256]
        // ==================================================
        CudaTimer timer_lm_head;
        timer_lm_head.start();
        launch_gemm_2048x128256(
            final_norm_output_.data_bf16(),
            weights_.lm_head,
            logits_.data_bf16(),
            1,
            llama::VOCAB_SIZE,
            llama::HIDDEN_SIZE
        );
        cudaDeviceSynchronize();
        performance_stats_.prefill_lm_head_ms =
            timer_lm_head.stop_ms();
        // ==================================================
        // 8. ARGMAX / SAMPLING
        // ==================================================
        int next_token = 0;
        int* d_next_token = nullptr;
        cudaMalloc(
            &d_next_token,
            sizeof(int)
        );
        CudaTimer timer_sampling;
        timer_sampling.start();
        launch_argmax(
            logits_.data_bf16(),
            d_next_token,
            0
        );
        cudaMemcpy(
            &next_token,
            d_next_token,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
        cudaDeviceSynchronize();
        performance_stats_.prefill_sampling_ms =
            timer_sampling.stop_ms();
        cudaFree(d_next_token);
        // ==================================================
        // 9. Cleanup
        // ==================================================
        cudaFree(prompt_tokens_gpu);
        // ==================================================
        // 10. TOTAL PREFILL TIME
        // ==================================================
        performance_stats_.prefill_ms =
            total_timer.stop_ms();
        return next_token;
    }

LlamaRuntime::LlamaRuntime(
    const std::string& weight_path,
    const std::string& tokenizer_path
)
{
    // Load tokenizer
    tokenizer_ = std::make_unique<transformer::tokenizer::Tokenizer>(
        tokenizer_path
    );
    // Load weights
    llama::load_weights(
        weight_path.c_str(),
        weights_
    );
    // Initialize runtime
    initialize();
}
LlamaRuntime::~LlamaRuntime(){
    // Free RoPE tables
    if(cos_table_){
        cudaFree(cos_table_);
        cos_table_ = nullptr;
    }
    if(sin_table_){
        cudaFree(sin_table_);
        sin_table_ = nullptr;
    }
    // Free token buffer
    if(token_buffer_){
        cudaFree(token_buffer_);
        token_buffer_ = nullptr;
    }
    // Free model weights
    llama::free_weights(
        weights_
    );
}
void LlamaRuntime::initialize(){
    // ==========================================
    // Create RoPE tables
    // ==========================================
    build_rope_tables();
    // ==========================================
    // Create Transformer
    // ==========================================
    transformer_ =
        std::make_unique<Transformer>(
            weights_,
            cos_table_,
            sin_table_,
            max_sequence_length_
        );
    // ==========================================
    // Allocate hidden state buffer
    // ==========================================
    hidden_states_.allocate({
        max_sequence_length_,
        llama::HIDDEN_SIZE
    }, DataType::BF16);

    transformer_output_.allocate({
        max_sequence_length_,
        llama::HIDDEN_SIZE
    }, DataType::BF16);
    final_norm_output_.allocate({
        1,
        llama::HIDDEN_SIZE
    }, DataType::BF16);
    
    last_hidden_state_.allocate({
        1,
        llama::HIDDEN_SIZE
    },
    DataType::BF16);
    // ==========================================
    // Allocate LM head output buffer
    // ==========================================
    logits_.allocate({
            1,
            llama::VOCAB_SIZE
        },
        DataType::BF16
    );
    // ==========================================
    // Allocate token input buffer
    // ==========================================
    cudaMalloc(
        &token_buffer_,
        sizeof(int)
    );
    printf(
        "[Runtime] Initialization complete\n"
    );
}
} // namespace runtime