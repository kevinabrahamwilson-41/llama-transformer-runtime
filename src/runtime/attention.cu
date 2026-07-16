/*
attention.cu
│
├── RMSNorm kernel
├── GEMM 2048×2048
├── GEMM 2048×512
├── RoPE
├── GQA
├── FlashAttention
└── GEMM 2048×2048
*/
#include "attention.hpp"
#include "tensor.hpp"
#include "weights.hpp"
#include <cuda_runtime.h>
#include "../kernels/rmsnorm/rmsnorm.hpp"
#include "../kernels/gemm/gemm_2048x2048.hpp"
#include "../kernels/gemm/gemm_2048x512.hpp"
#include "../kernels/rope/rope.hpp"
#include "../kernels/flashattention/flashattention.h"
#include "layout.hpp"
namespace runtime
{
static void debug_bf16(
    const char* name,
    const __nv_bfloat16* ptr,
    int elements
)
{
    std::vector<__nv_bfloat16> host(
        elements
    );

    cudaMemcpy(
        host.data(),
        ptr,
        elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );

    float max_abs = 0.0f;
    int max_idx = 0;

    for (int i = 0; i < elements; ++i)
    {
        float value =
            __bfloat162float(host[i]);

        if (fabsf(value) > max_abs)
        {
            max_abs = fabsf(value);
            max_idx = i;
        }
    }

    printf(
        "%s max_abs = %.6f at index %d\n",
        name,
        max_abs,
        max_idx
    );
}
Attention::Attention(
    __nv_bfloat16* input_norm,
    __nv_bfloat16* q_proj,
    __nv_bfloat16* k_proj,
    __nv_bfloat16* v_proj,
    __nv_bfloat16* o_proj,
    float* cos_table,
    float* sin_table
)
    : input_norm_(input_norm),
      q_proj_(q_proj),
      k_proj_(k_proj),
      v_proj_(v_proj),
      o_proj_(o_proj),
      cos_table_(cos_table),
      sin_table_(sin_table)
{
}

void Attention::forward(
    const Tensor& input,
    Tensor& output
) const
{
    int tokens =
        static_cast<int>(
            input.shape()[0]
        );

    // =========================================================
    // 1. RMSNorm
    //
    // [tokens, 2048]
    //        ↓
    // [tokens, 2048]
    // =========================================================

    Tensor normalized(
        {tokens, 2048},
        DataType::BF16
    );

    rmsnorm_launch<2048>(
        input.data_bf16(),
        input_norm_,
        normalized.data_bf16(),
        tokens,
        1e-5f
    );

    // =========================================================
    // 2. Q projection
    //
    // [tokens, 2048] × [2048, 2048]
    //        ↓
    // [tokens, 2048]
    // =========================================================

    Tensor q(
        {tokens, 2048},
        DataType::BF16
    );

    launch_gemm_2048x2048(
        normalized.data_bf16(),
        q_proj_,
        q.data_bf16(),
        tokens,
        2048,
        2048
    );
cudaDeviceSynchronize();

cudaError_t err = cudaGetLastError();

if (err != cudaSuccess)
{
    printf(
        "Q GEMM ERROR: %s\n",
        cudaGetErrorString(err)
    );
}

    debug_bf16(
        "Q",
        q.data_bf16(),
        tokens * 2048
    );

    // =========================================================
    // 3. K projection
    //
    // [tokens, 2048] × [2048, 512]
    //        ↓
    // [tokens, 512]
    // =========================================================

    Tensor k(
        {tokens, 512},
        DataType::BF16
    );

    launch_gemm_2048x512(
        normalized.data_bf16(),
        k_proj_,
        k.data_bf16(),
        tokens,
        512,
        2048
    );

    // =========================================================
    // 4. V projection
    //
    // [tokens, 2048] × [2048, 512]
    //        ↓
    // [tokens, 512]
    // =========================================================

    Tensor v(
        {tokens, 512},
        DataType::BF16
    );

    launch_gemm_2048x512(
        normalized.data_bf16(),
        v_proj_,
        v.data_bf16(),
        tokens,
        512,
        2048
    );
    cudaDeviceSynchronize();

    debug_bf16(
        "V",
        v.data_bf16(),
        tokens * 512
    );

    // =========================================================
    // 5. RoPE + layout conversion
    //
    // Q:
    // [tokens, 2048]
    //        ↓
    // [32, tokens, 64]
    //
    // K:
    // [tokens, 512]
    //        ↓
    // [8, tokens, 64]
    //
    // V:
    // [tokens, 512]
    //        ↓
    // [8, tokens, 64]
    // =========================================================

    Tensor q_flash(
        {32, tokens, 64},
        DataType::BF16
    );

    Tensor k_flash(
        {8, tokens, 64},
        DataType::BF16
    );

    Tensor v_flash(
        {8, tokens, 64},
        DataType::BF16
    );

    launch_rope_qkv(
        q.data_bf16(),
        q_flash.data_bf16(),

        k.data_bf16(),
        k_flash.data_bf16(),

        v.data_bf16(),
        v_flash.data_bf16(),

        cos_table_,
        sin_table_,
        tokens
    );
    cudaDeviceSynchronize();

    debug_bf16(
        "Q_FLASH",
        q_flash.data_bf16(),
        32 * tokens * 64
    );

    debug_bf16(
        "K_FLASH",
        k_flash.data_bf16(),
        8 * tokens * 64
    );

    debug_bf16(
        "V_FLASH",
        v_flash.data_bf16(),
        8 * tokens * 64
    );
    // =========================================================
    // 6. FlashAttention
    //
    // Q: [32, tokens, 64]
    // K: [8,  tokens, 64]
    // V: [8,  tokens, 64]
    //
    // O: [32, tokens, 64]
    // =========================================================

    Tensor flash_output(
        {32, tokens, 64},
        DataType::BF16
    );
    transformer::FlashAttentionParams flash_params{};

        flash_params.Q =
            q_flash.data_bf16();

        flash_params.K =
            k_flash.data_bf16();

        flash_params.V =
            v_flash.data_bf16();

        flash_params.O =
            flash_output.data_bf16();

        flash_params.L =
            nullptr;

        flash_params.batch_size =
            1;

        flash_params.num_heads =
            32;

        flash_params.num_kv_heads =
            8;

        flash_params.seq_len =
            tokens;

        flash_params.d_head =
            64;

        flash_params.scale =
            1.0f / sqrtf(64.0f);

        flash_params.causal =
            true;

        flash_params.dtype =
            transformer::DType::BF16;

        flash_params.stream =
            0;

        transformer::launch_flash_attention(
            flash_params
        );
        cudaDeviceSynchronize();

        debug_bf16(
            "FLASH_OUTPUT",
            flash_output.data_bf16(),
            32 * tokens * 64
        );
    // =========================================================
    // 7. Flash output layout conversion
    //
    // FlashAttention output:
    // [32, tokens, 64]
    //
    // Convert to GEMM layout:
    // [tokens, 2048]
    // =========================================================

    Tensor attention_input(
        {tokens, 2048},
        DataType::BF16
    );

    launch_flash_output_layout(
        flash_output.data_bf16(),
        attention_input.data_bf16(),
        tokens
    );
    cudaDeviceSynchronize();

debug_bf16(
    "ATTENTION_INPUT",
    attention_input.data_bf16(),
    tokens * 2048
);
    // =========================================================
    // 8. Output projection
    //
    // [tokens, 2048] × [2048, 2048]
    //        ↓
    // [tokens, 2048]
    // =========================================================

    launch_gemm_2048x2048(
        attention_input.data_bf16(),
        o_proj_,
        output.data_bf16(),
        tokens,
        2048,
        2048
    );
    cudaDeviceSynchronize();

    debug_bf16(
        "K",
        k.data_bf16(),
        tokens * 512
    );
}
}