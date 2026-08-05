#include "llama_runtime.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "../kernels/embedding/embedding.hpp"
#include "../kernels/rmsnorm/rmsnorm.hpp"
#include "../kernels/gemm/gemm_2048x128256.hpp"
#include "../kernels/sampling/argmax_128256.hpp"
#include <utility>
namespace runtime{
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
    std::string LlamaRuntime::generate(
        const std::string& prompt,
        int max_new_tokens,
        bool use_topk,
        int k
    ){
        // ==================================================
        // 1. Encode prompt
        // ==================================================
        transformer::tokenizer::TokenSequence tokens =
            tokenizer_->encode(
                prompt
            );
        transformer::tokenizer::TokenSequence generated_tokens;
        // Copy prompt tokens
        for(auto token : tokens){
            generated_tokens.push_back(token);
        }
        // ==================================================
        // 2. Prefill
        // ==================================================
        int position = 0;
        int next_token = 0;
        for(auto token_id : tokens){
            next_token =
                forward_next_token(
                    token_id,
                    position
                );
            position++;
        }
        // ==================================================
        // 3. Generate new tokens
        // ==================================================
        for(int i = 0; i < max_new_tokens; i++){
            int token =
                forward_next_token(
                    next_token,
                    position
                );
            generated_tokens.push_back(
                token
            );
            next_token = token;
            position++;
            // EOS check
            if(token == tokenizer_->eos_id()){
                break;
            }
        }
        // ==================================================
        // 4. Decode
        // ==================================================
        return tokenizer_->decode(
            generated_tokens
        );
    }

    void dump_tensor(
        const char* path,
        const __nv_bfloat16* device_tensor,
        int elements
    )
    {
        std::vector<__nv_bfloat16> host(elements);

        cudaMemcpy(
            host.data(),
            device_tensor,
            elements * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost
        );


        FILE* file = fopen(path, "w");

        if(!file)
        {
            printf("Failed to open dump file\n");
            exit(EXIT_FAILURE);
        }


        for(int i = 0; i < elements; i++)
        {
            fprintf(
                file,
                "%.9g\n",
                __bfloat162float(host[i])
            );
        }

        fclose(file);
    }
    int LlamaRuntime::forward_next_token(int token_id,int position){
                // ==================================================
        // 1. Copy token id to GPU
        // ==================================================
        cudaMemcpy(
            token_buffer_,
            &token_id,
            sizeof(int),
            cudaMemcpyHostToDevice
        );
        // ==================================================
        // 2. Token embedding
        //
        // token_id
        //     |
        //     ↓
        // hidden_states [1,2048]
        // ==================================================

        launch_embedding(
            token_buffer_,
            weights_.embed_tokens,
            hidden_states_.data_bf16(),
            1
        );


        // ==================================================
        // 3. Transformer forward
        //
        // hidden_states
        //     |
        //     ↓
        // 16 Transformer blocks
        // ==================================================
        transformer_->forward(
            hidden_states_,
            transformer_output_,
            position
        );
        dump_tensor(
            "/tmp/before_norm_full.txt",
            transformer_output_.data_bf16(),
            2048
        );
        // ==================================================
        // 4. Final RMSNorm
        // ==================================================
        // BEFORE RMSNorm
        dump_tensor(
            "/tmp/cuda_before_final_norm.txt",
            transformer_output_.data_bf16(),
            llama::HIDDEN_SIZE
        );
        rmsnorm_launch<llama::HIDDEN_SIZE>(
            transformer_output_.data_bf16(),
            weights_.final_norm,
            hidden_states_.data_bf16(),
            1,
            1e-5f
        );
        cudaDeviceSynchronize();
        dump_tensor(
            "/tmp/cuda_final_norm.txt",
            hidden_states_.data_bf16(),
            llama::HIDDEN_SIZE
        );
        // ==================================================
        // 5. LM Head
        //
        // [1,2048]
        //      x
        // [2048,128256]
        //
        // = logits
        // ==================================================
        launch_gemm_2048x128256(
            hidden_states_.data_bf16(),
            weights_.lm_head,
            logits_.data_bf16(),
            1,                  // M
            llama::VOCAB_SIZE,  // N
            llama::HIDDEN_SIZE  // K
        );
        // ==================================================
        // DEBUG LM HEAD OUTPUT
        // ==================================================

        cudaDeviceSynchronize();

        std::vector<__nv_bfloat16> h_logits(
            llama::VOCAB_SIZE
        );

        cudaMemcpy(
            h_logits.data(),
            logits_.data_bf16(),
            llama::VOCAB_SIZE * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost
        );


        float max_val = -1e9f;
        int max_idx = -1;


        for(int i = 0; i < llama::VOCAB_SIZE; i++)
        {
            float v = __bfloat162float(h_logits[i]);

            if(v > max_val)
            {
                max_val = v;
                max_idx = i;
            }
        }
        // ==================================================
        // 6. Argmax sampling
        // ==================================================
        int next_token = 0;
        launch_argmax(
            logits_.data_bf16(),
            &next_token,
            0
        );
        cudaDeviceSynchronize();
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
LlamaRuntime::LlamaRuntime(){
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
        1,
        llama::HIDDEN_SIZE
    },
    DataType::BF16
    );
    transformer_output_.allocate({
        1,
        llama::HIDDEN_SIZE
    },
    DataType::BF16
    );
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