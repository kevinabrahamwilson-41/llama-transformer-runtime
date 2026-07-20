#include "transformer_block.hpp"
#include "attention.hpp"
#include "ffn.hpp"
#include "tensor.hpp"
#include "weights.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

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
static void dump_tensor(
    const char* path,
    const __nv_bfloat16* tensor,
    int elements
)
{
    FILE* file = std::fopen(path, "w");

    if (!file)
    {
        std::printf(
            "Failed to open dump file: %s\n",
            path
        );

        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < elements; ++i)
    {
        float value =
            __bfloat162float(
                tensor[i]
            );

        std::fprintf(
            file,
            "%.9g\n",
            value
        );
    }

    std::fclose(file);

    std::printf(
        "[TEST] Dumped tensor: %s\n",
        path
    );
}
int main(){
    constexpr int TOKENS = 128;
    constexpr int HIDDEN = 2048;
    constexpr int ROTARY_DIM = 32;
    constexpr int HEAD_DIM = 64;
    constexpr float ROPE_THETA = 500000.0f;
    constexpr float ROPE_FACTOR = 32.0f;
    constexpr float LOW_FREQ_FACTOR = 1.0f;
    constexpr float HIGH_FREQ_FACTOR = 4.0f;
    constexpr float ORIGINAL_CONTEXT_LENGTH = 8192.0f;

    const char* WEIGHTS_PATH =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama_weights.bin";


    printf(
        "============================================================\n"
    );

    printf(
        "        LLAMA 3.2 1B TRANSFORMER BLOCK[0] TEST\n"
    );

    printf(
        "============================================================\n\n"
    );

    printf(
        "Tokens : %d\n",
        TOKENS
    );

    printf(
        "Hidden : %d\n\n",
        HIDDEN
    );


    // ========================================================
    // 1. LOAD REAL MODEL WEIGHTS
    // ========================================================

    printf(
        "[TEST] Loading model weights...\n"
    );


    llama::LlamaWeights weights{};

    llama::load_weights(
        WEIGHTS_PATH,
        weights
    );
    // TEMP: dump K projection weights

constexpr int K_ELEMENTS = 2048 * 512;

std::vector<__nv_bfloat16> host_k(
    K_ELEMENTS
);


CUDA_CHECK(
    cudaMemcpy(
        host_k.data(),
        weights.layers[0].k_proj,
        K_ELEMENTS * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    )
);


FILE* f = fopen(
    "/tmp/k_weight_dump.txt",
    "w"
);


if(!f)
{
    printf("Failed opening k dump\n");
    exit(1);
}


for(int i = 0; i < K_ELEMENTS; i++)
{
    fprintf(
        f,
        "%.9g\n",
        __bfloat162float(
            host_k[i]
        )
    );
}


fclose(f);


printf(
    "[TEST] K weight dumped.\n"
);

constexpr int V_ELEMENTS = 2048 * 512;

std::vector<__nv_bfloat16> host_v(
    V_ELEMENTS
);

CUDA_CHECK(
    cudaMemcpy(
        host_v.data(),
        weights.layers[0].v_proj,
        V_ELEMENTS * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    )
);


FILE* fv = fopen(
    "/tmp/v_weight_dump.txt",
    "w"
);


for(int i = 0; i < V_ELEMENTS; i++)
{
    fprintf(
        fv,
        "%.9g\n",
        __bfloat162float(host_v[i])
    );
}


fclose(fv);


printf(
    "[TEST] V weight dumped.\n"
);
    printf(
        "[TEST] Real weights loaded.\n\n"
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


    // ========================================================
    // 3. CONSTRUCT ATTENTION USING LAYER 0 WEIGHTS
    // ========================================================

    printf(
        "[TEST] Constructing Attention using layer 0...\n"
    );


    runtime::Attention attention(
        weights.layers[0].input_layernorm,
        weights.layers[0].q_proj,
        weights.layers[0].k_proj,
        weights.layers[0].v_proj,
        weights.layers[0].o_proj,
        cos_table,
        sin_table
    );


    // ========================================================
    // 4. CONSTRUCT FFN USING LAYER 0 WEIGHTS
    // ========================================================

    printf(
        "[TEST] Constructing FeedForward using layer 0...\n"
    );


    runtime::FeedForward ffn(
        weights.layers[0].post_attention_layernorm,
        weights.layers[0].gate_proj,
        weights.layers[0].up_proj,
        weights.layers[0].down_proj
    );


    // ========================================================
    // 5. CONSTRUCT TRANSFORMER BLOCK 0
    // ========================================================

    printf(
        "[TEST] Constructing TransformerBlock[0]...\n"
    );


    runtime::TransformerBlock block(
        attention,
        ffn
    );


    // ========================================================
    // 6. CREATE INPUT
    // ========================================================

    runtime::Tensor input(
        {TOKENS, HIDDEN},
        runtime::DataType::BF16
    );


    runtime::Tensor output(
        {TOKENS, HIDDEN},
        runtime::DataType::BF16
    );


    // ========================================================
    // 7. INITIALIZE DETERMINISTIC INPUT
    // ========================================================

    std::vector<__nv_bfloat16> host_input(
        TOKENS * HIDDEN
    );


    for (int i = 0; i < TOKENS * HIDDEN; ++i)
    {
        float value =
            1.0f +
            static_cast<float>(i % 17);


        host_input[i] =
            __float2bfloat16(value);
    }


    CUDA_CHECK(
        cudaMemcpy(
            input.data_bf16(),
            host_input.data(),
            host_input.size() *
            sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        )
    );


    printf(
        "[TEST] Input initialized.\n\n"
    );


    // ========================================================
    // 8. RUN TRANSFORMER BLOCK[0]
    // ========================================================

    printf(
        "============================================================\n"
    );

    printf(
        "                 RUNNING BLOCK[0]\n"
    );

    printf(
        "============================================================\n\n"
    );


    block.forward(
        input,
        output
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    printf(
        "\n[TEST] TransformerBlock[0] complete.\n\n"
    );


    // ========================================================
    // 9. COPY OUTPUT BACK
    // ========================================================

    std::vector<__nv_bfloat16> host_output(
        TOKENS * HIDDEN
    );


    CUDA_CHECK(
        cudaMemcpy(
            host_output.data(),
            output.data_bf16(),
            host_output.size() *
            sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost
        )
    );
    dump_tensor(
        "/tmp/cuda_block0_output.txt",
        host_output.data(),
        TOKENS * HIDDEN
    );

    // ========================================================
    // 10. BASIC OUTPUT VALIDATION
    // ========================================================

    int nan_count = 0;
    int inf_count = 0;

    float max_abs = 0.0f;


    for (int i = 0; i < TOKENS * HIDDEN; ++i)
    {
        float value =
            __bfloat162float(
                host_output[i]
            );


        if (std::isnan(value))
        {
            nan_count++;
        }


        if (std::isinf(value))
        {
            inf_count++;
        }


        max_abs =
            std::max(
                max_abs,
                std::fabs(value)
            );
    }


    // ========================================================
    // RESULT
    // ========================================================

    printf(
        "============================================================\n"
    );

    printf(
        "                         RESULT\n"
    );

    printf(
        "============================================================\n\n"
    );


    printf(
        "Output max_abs : %.6f\n",
        max_abs
    );


    printf(
        "NaN count      : %d\n",
        nan_count
    );


    printf(
        "Inf count      : %d\n\n",
        inf_count
    );


    if (nan_count == 0 && inf_count == 0)
    {
        printf(
            "SUCCESS\n"
        );

        printf(
            "TransformerBlock[0] executed with REAL Llama 3.2 1B weights.\n"
        );

        printf(
            "Attention + FFN + residual flow completed.\n"
        );
    }
    else
    {
        printf(
            "FAIL\n"
        );

        printf(
            "Invalid numerical output detected.\n"
        );
    }


    // ========================================================
    // CLEANUP
    // ========================================================

    cudaFree(
        cos_table
    );


    cudaFree(
        sin_table
    );


    llama::free_weights(
        weights
    );


    return (
        nan_count == 0 &&
        inf_count == 0
    )
        ? 0
        : 1;

}
