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
#include "../kernels/embedding/embedding.hpp"
#include "../kernels/rmsnorm/rmsnorm.hpp"
#include "../kernels/gemm/gemm_2048x128256.hpp"
#include "../kernels/sampling/argmax_128256.hpp"
namespace fs = std::filesystem;
namespace runtime{
    static void save_int_checkpoint(
    const std::string& name,
    const int* device_ptr,
    size_t elements
){
    namespace fs = std::filesystem;

    fs::create_directories("cuda_debug");

    std::vector<int> host(elements);

    cudaMemcpy(
        host.data(),
        device_ptr,
        elements * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    std::ofstream file(
        "cuda_debug/" + name + ".bin",
        std::ios::binary
    );

    uint64_t count =
        static_cast<uint64_t>(elements);

    uint32_t dtype = 1;

    file.write(
        reinterpret_cast<const char*>(&count),
        sizeof(count)
    );

    file.write(
        reinterpret_cast<const char*>(&dtype),
        sizeof(dtype)
    );

    file.write(
        reinterpret_cast<const char*>(host.data()),
        elements * sizeof(int)
    );

    file.close();

    printf(
        "[CHECKPOINT] SAVED: cuda_debug/%s.bin "
        "elements=%zu\n",
        name.c_str(),
        elements
    );
}
    // ============================================================
    // CUDA DEBUG CHECKPOINTS
    // ============================================================
    static constexpr const char* DEBUG_DIR = "cpp_debug";
    // ============================================================
    // Save BF16 GPU tensor as FLOAT32 binary checkpoint
    // ============================================================
    static void save_checkpoint(
    const std::string& name,
    const __nv_bfloat16* device_ptr,
    size_t elements
){
    namespace fs = std::filesystem;

    const std::string DEBUG_DIR = "cuda_debug";

    fs::create_directories(DEBUG_DIR);

    if(device_ptr == nullptr){
        printf(
            "[CHECKPOINT] %s: NULL\n",
            name.c_str()
        );
        return;
    }

    // --------------------------------------------------
    // Copy BF16 GPU data -> CPU
    // --------------------------------------------------

    std::vector<__nv_bfloat16> host_bf16(elements);

    cudaError_t err = cudaMemcpy(
        host_bf16.data(),
        device_ptr,
        elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );

    if(err != cudaSuccess){
        printf(
            "[CHECKPOINT] %s: cudaMemcpy FAILED: %s\n",
            name.c_str(),
            cudaGetErrorString(err)
        );
        return;
    }

    // --------------------------------------------------
    // Convert BF16 -> FP32
    // --------------------------------------------------

    std::vector<float> host_float(elements);

    for(size_t i = 0; i < elements; i++){
        host_float[i] =
            __bfloat162float(host_bf16[i]);
    }

    // --------------------------------------------------
    // Save binary
    // --------------------------------------------------

    const std::string path =
        DEBUG_DIR + "/" + name + ".bin";

    std::ofstream file(
        path,
        std::ios::binary
    );

    if(!file){
        printf(
            "[CHECKPOINT] %s: FAILED OPENING FILE\n",
            name.c_str()
        );
        return;
    }

    // Number of elements
    uint64_t count =
        static_cast<uint64_t>(elements);

    // Always float32 on disk
    uint32_t dtype = 0;

    file.write(
        reinterpret_cast<const char*>(&count),
        sizeof(count)
    );

    file.write(
        reinterpret_cast<const char*>(&dtype),
        sizeof(dtype)
    );

    file.write(
        reinterpret_cast<const char*>(host_float.data()),
        elements * sizeof(float)
    );

    file.close();
    printf(
        "[LLAMA_RUNTIME_CU] SAVED: %s "
        "elements=%zu\n",
        path.c_str(),
        elements
    );
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
    
    static void print_first32(
        const char* name,
        const __nv_bfloat16* device_ptr,
        int elements
    ){
        int n = (elements < 10) ? elements : 10;
        std::vector<__nv_bfloat16> host(n);
        cudaMemcpy(
            host.data(),
            device_ptr,
            n * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost
        );
        printf("\n==============================\n");
        printf("%s\n", name);
        printf("==============================\n");
        for (int i = 0; i < n; i++){
            printf("%.9f\n", __bfloat162float(host[i]));
        }
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
        int next_token =
            forward_prefill(tokens);
        // All prompt tokens are now already in the KV cache.
        int position = tokens.size();
        // ==================================================
        // 3. Generate new tokens
        // ==================================================
        for(int i = 0; i < max_new_tokens; i++){
            int token =
                decode_forward(
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
    int LlamaRuntime::decode_forward(int token_id,int position){
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
        cudaDeviceSynchronize();
        save_checkpoint(
            "generation_embedding",
            hidden_states_.data_bf16(),
            llama::HIDDEN_SIZE
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
            position,
            1
        );
        // ==================================================
        // 4. Final RMSNorm
        // ==================================================
        rmsnorm_launch<llama::HIDDEN_SIZE>(
            transformer_output_.data_bf16(),
            weights_.final_norm,
            final_norm_output_.data_bf16(),
            1,
            1e-5f
        );
        cudaDeviceSynchronize();
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
            final_norm_output_.data_bf16(),
            weights_.lm_head,
            logits_.data_bf16(),
            1,                  // M
            llama::VOCAB_SIZE,  // N
            llama::HIDDEN_SIZE  // K
        );
        // Argmax kernel
        int next_token = 0;
        launch_argmax(
            logits_.data_bf16(),
            &next_token,
            0
        );
        cudaDeviceSynchronize();
        printf(
            "GPU ARGMAX TOKEN = %d\n",
            next_token
        );
        return next_token;
    }
    int LlamaRuntime::forward_prefill(
        const transformer::tokenizer::TokenSequence& tokens
    ){
        int seq_len = tokens.size();
        // ==================================================
        // 1. Copy prompt tokens to GPU
        // ==================================================
        int* prompt_tokens_gpu;
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
        save_int_checkpoint(
            "input_ids",
            prompt_tokens_gpu,
            seq_len
        );
        // ==================================================
        // 2. Embedding
        //
        // tokens
        //    |
        //    v
        // hidden_states [seq_len,2048]
        // ==================================================
        launch_embedding(
            prompt_tokens_gpu,
            weights_.embed_tokens,
            hidden_states_.data_bf16(),
            seq_len
        );
        cudaDeviceSynchronize();
        save_checkpoint(
            "embedding",
            hidden_states_.data_bf16(),
            static_cast<size_t>(seq_len) * llama::HIDDEN_SIZE
        );
        // ==================================================
        // 3. Transformer PREFILL
        // IMPORTANT:
        // This is what fills KV cache.
        // position = 0
        // seq_len  = prompt length
        // Example:
        // 22 prompt tokens:
        // Q = [32,22,64]
        // K = [8,22,64]
        // V = [8,22,64]
        // cache positions:
        // 0 ... 21
        // ==================================================

        transformer_->forward(
            hidden_states_,
            transformer_output_,
            0,
            seq_len
        );
        cudaDeviceSynchronize();
        // ==================================================
        // DEBUG
        // ==================================================
        printf("\n==============================\n");
        printf("PREFILL TRANSFORMER COMPLETE\n");
        printf("seq_len=%d\n", seq_len);
        printf("==============================\n");
        // ==================================================
        // 4. Extract LAST TOKEN hidden state
        // transformer_output:
        // [seq_len,2048]
        // We only need:
        // [last token,2048]
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
        // 5. Final RMSNorm
        // [1,2048]
        //       |
        //       v
        // [1,2048]
        // ==================================================
        rmsnorm_launch<llama::HIDDEN_SIZE>(
            last_hidden_state_.data_bf16(),
            weights_.final_norm,
            final_norm_output_.data_bf16(),
            1,
            1e-5f
        );
        cudaDeviceSynchronize();
        // ==================================================
        // 6. LM Head
        // [1,2048]
        //      x
        // [2048,128256]
        //      |
        //      v
        // logits [1,128256]
        // ==================================================
        launch_gemm_2048x128256(
            final_norm_output_.data_bf16(),
            weights_.lm_head,
            logits_.data_bf16(),
            1,
            llama::VOCAB_SIZE,
            llama::HIDDEN_SIZE
        );
        cudaDeviceSynchronize();
        // ==================================================
        // 7. Argmax
        // ==================================================
        int next_token = 0;
        launch_argmax(
            logits_.data_bf16(),
            &next_token,
            0
        );
        cudaDeviceSynchronize();
        // ==================================================
        // Cleanup
        // ==================================================
        cudaFree(prompt_tokens_gpu);
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