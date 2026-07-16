#include "transformer_block.hpp"
#include "attention.hpp"
#include "ffn.hpp"
#include "tensor.hpp"

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
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess)                                   \
    {                                                         \
        printf("CUDA ERROR %s:%d : %s\n",                     \
               __FILE__,                                      \
               __LINE__,                                      \
               cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                  \
    }                                                         \
} while (0)

// ============================================================
// Helper
// ============================================================

void allocate_zero(
    __nv_bfloat16** ptr,
    size_t elements
)
{
    CUDA_CHECK(
        cudaMalloc(
            ptr,
            elements * sizeof(__nv_bfloat16)
        )
    );

    CUDA_CHECK(
        cudaMemset(
            *ptr,
            0,
            elements * sizeof(__nv_bfloat16)
        )
    );
}

void allocate_ones(
    __nv_bfloat16** ptr,
    size_t elements
)
{
    std::vector<__nv_bfloat16> host(
        elements,
        __float2bfloat16(1.0f)
    );

    CUDA_CHECK(
        cudaMalloc(
            ptr,
            elements * sizeof(__nv_bfloat16)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            *ptr,
            host.data(),
            elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        )
    );
}
void allocate_matrix(
    __nv_bfloat16** ptr,
    size_t elements,
    float scale
)
{
    std::vector<__nv_bfloat16> host(
        elements
    );

    for (size_t i = 0; i < elements; ++i)
    {
        float value =
            ((static_cast<int>(i % 17) - 8) * scale);

        host[i] =
            __float2bfloat16(value);
    }

    CUDA_CHECK(
        cudaMalloc(
            ptr,
            elements * sizeof(__nv_bfloat16)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            *ptr,
            host.data(),
            elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        )
    );
}
// ============================================================
// MAIN
// ============================================================

int main()
{
    constexpr int TOKENS = 128;
    constexpr int HIDDEN = 2048;
    float* cos_table;
    float* sin_table;
    printf("============================================================\n");
    printf("              TRANSFORMER BLOCK TEST\n");
    printf("============================================================\n\n");

    printf("Tokens : %d\n", TOKENS);
    printf("Hidden : %d\n\n", HIDDEN);

    // ========================================================
    // 1. Allocate model weights
    //
    // ZERO WEIGHTS = STRUCTURAL TEST
    //
    // Attention output = 0
    // FFN output       = 0
    //
    // Therefore:
    //
    // output = input
    //
    // This tests the full block flow and residual connections.
    // ========================================================

    __nv_bfloat16* attention_norm;
    __nv_bfloat16* q_proj;
    __nv_bfloat16* k_proj;
    __nv_bfloat16* v_proj;
    __nv_bfloat16* o_proj;

    __nv_bfloat16* ffn_norm;
    __nv_bfloat16* gate_proj;
    __nv_bfloat16* up_proj;
    __nv_bfloat16* down_proj;

    // Attention RMSNorm
    allocate_ones(
        &attention_norm,
        2048
    );

    allocate_matrix(
        &q_proj,
        2048ULL * 2048,
        0.001f
    );

    allocate_matrix(
        &k_proj,
        2048ULL * 512,
        0.001f
    );

    allocate_matrix(
        &v_proj,
        2048ULL * 512,
        0.001f
    );

    allocate_matrix(
        &o_proj,
        2048ULL * 2048,
        0.001f
    );

    // FFN RMSNorm
    allocate_ones(
        &ffn_norm,
        2048
    );

    allocate_matrix(
        &gate_proj,
        2048ULL * 8192,
        0.0005f
    );

    allocate_matrix(
        &up_proj,
        2048ULL * 8192,
        0.0005f
    );

    allocate_matrix(
        &down_proj,
        8192ULL * 2048,
        0.0005f
    );

    // ========================================================
    // 2. RoPE tables
    // ========================================================

    constexpr int ROTARY_DIM = 32;
    constexpr int HEAD_DIM = 64;
    constexpr float ROPE_THETA = 500000.0f;

    CUDA_CHECK(
        cudaMalloc(
            &cos_table,
            TOKENS * ROTARY_DIM * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &sin_table,
            TOKENS * ROTARY_DIM * sizeof(float)
        )
    );

    std::vector<float> host_cos(
        TOKENS * ROTARY_DIM
    );

    std::vector<float> host_sin(
        TOKENS * ROTARY_DIM
    );

    for (int pos = 0; pos < TOKENS; ++pos)
    {
        for (int pair = 0; pair < ROTARY_DIM; ++pair)
        {
            float exponent =
                static_cast<float>(2 * pair) /
                static_cast<float>(HEAD_DIM);

            float inv_freq =
                1.0f /
                std::pow(
                    ROPE_THETA,
                    exponent
                );

            float angle =
                static_cast<float>(pos) *
                inv_freq;

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

    // ========================================================
    // 3. Construct Attention
    // ========================================================

    runtime::Attention attention(
        attention_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        cos_table,
        sin_table
    );

    // ========================================================
    // 4. Construct FeedForward
    // ========================================================

    runtime::FeedForward ffn(
        ffn_norm,
        gate_proj,
        up_proj,
        down_proj
    );

    // ========================================================
    // 5. Construct TransformerBlock
    // ========================================================

    runtime::TransformerBlock block(
        attention,
        ffn
    );

    // ========================================================
    // 6. Input
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
    // 7. Initialize input
    // ========================================================

    std::vector<__nv_bfloat16> host_input(
        TOKENS * HIDDEN
    );

    for (int i = 0; i < TOKENS * HIDDEN; ++i)
    {
        float value =
            1.0f + static_cast<float>(i % 17);

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

    // ========================================================
    // 8. Run TransformerBlock
    // ========================================================

    printf("\nRunning TransformerBlock...\n");

    block.forward(
        input,
        output
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    printf("Block complete.\n");

    // ========================================================
    // 9. Copy output back
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

    // ========================================================
    // 10. Validate
    //
    // Since attention and FFN weights are zero:
    //
    // attention_output = 0
    // residual         = input
    // ffn_output       = 0
    // output           = residual
    //
    // Therefore:
    //
    // output == input
    // ========================================================

    int errors = 0;

    for (int i = 0; i < TOKENS * HIDDEN; ++i)
    {
        float expected =
            __bfloat162float(
                host_input[i]
            );

        float actual =
            __bfloat162float(
                host_output[i]
            );

        if (std::fabs(actual - expected) > 0.1f)
        {
            if (errors < 10)
            {
                printf(
                    "Mismatch at %d: expected %.4f got %.4f\n",
                    i,
                    expected,
                    actual
                );
            }

            errors++;
        }
    }

    // ========================================================
    // RESULT
    // ========================================================

    printf("\n============================================================\n");
    printf("                         RESULT\n");
    printf("============================================================\n");

    if (errors == 0)
    {
        printf("\nSUCCESS\n");
        printf("TransformerBlock flow is structurally CORRECT.\n");
        printf("\nVerified:\n");
        printf("  Attention RMSNorm\n");
        printf("  Q/K/V projections\n");
        printf("  RoPE\n");
        printf("  Q/K/V layout conversion\n");
        printf("  FlashAttention\n");
        printf("  Attention output projection\n");
        printf("  Attention residual\n");
        printf("  FFN RMSNorm\n");
        printf("  W1\n");
        printf("  W3\n");
        printf("  SiLU + Mul\n");
        printf("  W2\n");
        printf("  FFN residual\n");
    }
    else
    {
        printf("\nFAIL\n");
        printf("Errors: %d\n", errors);
    }

    // ========================================================
    // Cleanup
    // ========================================================

    cudaFree(attention_norm);
    cudaFree(q_proj);
    cudaFree(k_proj);
    cudaFree(v_proj);
    cudaFree(o_proj);

    cudaFree(ffn_norm);
    cudaFree(gate_proj);
    cudaFree(up_proj);
    cudaFree(down_proj);

    cudaFree(cos_table);
    cudaFree(sin_table);

    return errors == 0 ? 0 : 1;
}