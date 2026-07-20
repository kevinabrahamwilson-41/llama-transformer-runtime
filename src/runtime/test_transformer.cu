#include "transformer.hpp"
#include "weights.hpp"
#include "tensor.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <vector>
#include <cstdio>
#include <cmath>
constexpr float LOW_FREQ_FACTOR = 1.0f;
constexpr float HIGH_FREQ_FACTOR = 4.0f;
constexpr float ORIGINAL_CONTEXT_LENGTH = 8192.0f;


// ============================================================
// CUDA CHECK
// ============================================================

#define CUDA_CHECK(call)                                      \
    do                                                      \
    {                                                       \
        cudaError_t err = (call);                            \
                                                            \
        if (err != cudaSuccess)                             \
        {                                                   \
            printf(                                         \
                "CUDA ERROR %s:%d : %s\n",               \
                __FILE__,                                   \
                __LINE__,                                   \
                cudaGetErrorString(err)                     \
            );                                              \
                                                            \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)
    

int main()
{
    constexpr int TOKENS = 128;
    constexpr int HIDDEN = 2048;

    constexpr int ROTARY_DIM = 32;
    constexpr int HEAD_DIM = 64;

    constexpr float ROPE_THETA = 500000.0f;


    const char* WEIGHTS_PATH =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama_weights.bin";


    printf(
        "LLAMA 3.2 1B FULL TRANSFORMER TEST\n"
    );


    // ==========================================
    // 1. LOAD WEIGHTS
    // ==========================================

    llama::LlamaWeights weights{};


    llama::load_weights(
        WEIGHTS_PATH,
        weights
    );

    // ========================================================
    // 2. CREATE RoPE TABLES
    // ========================================================

    float* cos_table = nullptr;
    float* sin_table = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &cos_table,
            TOKENS *
            ROTARY_DIM *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &sin_table,
            TOKENS *
            ROTARY_DIM *
            sizeof(float)
        )
    );


    std::vector<float> host_cos(
        TOKENS * ROTARY_DIM
    );


    std::vector<float> host_sin(
        TOKENS * ROTARY_DIM
    );


   for (int pair = 0; pair < ROTARY_DIM; ++pair){
        float exponent =
            static_cast<float>(2 * pair) /
            static_cast<float>(HEAD_DIM);

        float inv_freq =
            1.0f /
            std::pow(
                ROPE_THETA,
                exponent
            );

        float wavelen =
            2.0f *
            static_cast<float>(M_PI) /
            inv_freq;

        constexpr float FACTOR = 32.0f;
        constexpr float LOW_FREQ_FACTOR = 1.0f;
        constexpr float HIGH_FREQ_FACTOR = 4.0f;
        constexpr float OLD_CONTEXT_LEN = 8192.0f;

        float low_freq_wavelen =
            OLD_CONTEXT_LEN /
            LOW_FREQ_FACTOR;

        float high_freq_wavelen =
            OLD_CONTEXT_LEN /
            HIGH_FREQ_FACTOR;

        float inv_freq_llama;

        if (wavelen < high_freq_wavelen)
        {
            // High frequency: unchanged
            inv_freq_llama = inv_freq;
        }
        else if (wavelen <= low_freq_wavelen)
        {
            // Medium frequency: smooth interpolation
            float smooth_factor =
                (
                    OLD_CONTEXT_LEN / wavelen
                    - LOW_FREQ_FACTOR
                )
                /
                (
                    HIGH_FREQ_FACTOR
                    - LOW_FREQ_FACTOR
                );

            inv_freq_llama =
                (1.0f - smooth_factor)
                * (inv_freq / FACTOR)
                +
                smooth_factor * inv_freq;
        }
        else
        {
            // Low frequency: scaled
            inv_freq_llama =
                inv_freq / FACTOR;
        }

        for (int pos = 0; pos < TOKENS; ++pos)
        {
            float angle =
                static_cast<float>(pos)
                * inv_freq_llama;

            host_cos[
                pos * ROTARY_DIM + pair
            ] = std::cos(angle);

            host_sin[
                pos * ROTARY_DIM + pair
            ] = std::sin(angle);
        }
    }


    CUDA_CHECK(
        cudaMemcpy(
            cos_table,
            host_cos.data(),
            host_cos.size() * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            sin_table,
            host_sin.data(),
            host_sin.size() * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    printf(
        "[TEST] RoPE tables created.\n\n"
    );
    // ==========================================
    // 3. CREATE FULL TRANSFORMER
    // ==========================================

    runtime::Transformer model(
        weights,
        cos_table,
        sin_table
    );


    printf(
        "Transformer with 16 layers created\n"
    );


    // ==========================================
    // 4. CREATE INPUT
    // ==========================================

    runtime::Tensor input(
        {TOKENS,HIDDEN},
        runtime::DataType::BF16
    );


    runtime::Tensor output(
        {TOKENS,HIDDEN},
        runtime::DataType::BF16
    );


    // initialize input
    std::vector<__nv_bfloat16> host_input(
        TOKENS * HIDDEN
    );


    for(int i=0;i<TOKENS*HIDDEN;i++)
    {
        host_input[i] =
            __float2bfloat16(
                1.0f + (i%17)
            );
    }


    cudaMemcpy(
        input.data_bf16(),
        host_input.data(),
        host_input.size()*sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice
    );


    // ==========================================
    // 5. RUN ALL 16 BLOCKS
    // ==========================================


    model.forward(
        input,
        output
    );


    cudaDeviceSynchronize();


    std::vector<__nv_bfloat16> host_output(
        TOKENS*HIDDEN
    );


    cudaMemcpy(
        host_output.data(),
        output.data_bf16(),
        host_output.size()*sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );


    printf(
        "Output[0]=%f\n",
        __bfloat162float(host_output[0])
    );

    // ==========================================
    // 6. CLEANUP
    // ==========================================

    cudaFree(cos_table);
    cudaFree(sin_table);

    llama::free_weights(
        weights
    );


    return 0;
}