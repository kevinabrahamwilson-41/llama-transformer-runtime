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
#include <cmath>
#include <stdexcept>
#include <string>
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
#include <vector>
#include "debug_dump.hpp"
namespace runtime{
#define CUDA_CHECK(call)                                      \
do                                                            \
{                                                             \
    cudaError_t err = (call);                                \
                                                              \
    if (err != cudaSuccess)                                   \
    {                                                         \
        throw std::runtime_error(                            \
            std::string("CUDA error: ") +                    \
            cudaGetErrorString(err)                          \
        );                                                    \
    }                                                         \
} while (0)
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
    printf(
    "\n========== ATTENTION FORWARD ==========\n"
    );
    printf(
        "ATTENTION TOKENS = %d\n",
        tokens
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
    dump_bf16(
        "/tmp/cuda_attention_norm.txt",
        normalized.data_bf16(),
        tokens * 2048
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
    printf(
        "[ATTENTION] Q GEMM: M=%d N=2048 K=2048\n",
        tokens
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

    dump_bf16(
        "/tmp/cuda_q.txt",
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
    printf(
        "[ATTENTION] K GEMM: M=%d N=512 K=2048\n",
        tokens
    );

    launch_gemm_2048x512(
        normalized.data_bf16(),
        k_proj_,
        k.data_bf16(),
        tokens,
        512,
        2048
    );
    cudaDeviceSynchronize();

std::vector<__nv_bfloat16> h_k(tokens * 512);

cudaMemcpy(
    h_k.data(),
    k.data_bf16(),
    tokens * 512 * sizeof(__nv_bfloat16),
    cudaMemcpyDeviceToHost
);

printf("K GEMM first values:\n");

for(int i = 0; i < 10; i++)
{
    printf("%f\n",
        __bfloat162float(h_k[i])
    );
}

printf(
    "K[6556] = %f\n",
    __bfloat162float(h_k[6556])
);

cudaMemcpy(
    h_k.data(),
    k.data_bf16(),
    tokens * 512 * sizeof(__nv_bfloat16),
    cudaMemcpyDeviceToHost
);


printf("K GEMM first values:\n");

for(int i = 0; i < 10; i++)
{
    printf(
        "%f\n",
        __bfloat162float(h_k[i])
    );
}


printf(
    "K[6556] = %f\n",
    __bfloat162float(h_k[6556])
);
    dump_bf16(
        "/tmp/cuda_k.txt",
        k.data_bf16(),
        tokens * 512
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
    printf(
        "[ATTENTION] V GEMM: M=%d N=512 K=2048\n",
        tokens
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

    dump_bf16(
        "/tmp/cuda_v.txt",
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

    dump_bf16(
        "/tmp/cuda_q_rope.txt",
        q_flash.data_bf16(),
        32 * tokens * 64
    );

    dump_bf16(
        "/tmp/cuda_k_rope.txt",
        k_flash.data_bf16(),
        8 * tokens * 64
    );

    dump_bf16(
        "/tmp/cuda_v_rope.txt",
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

        printf(
            "FLASH PARAMS: causal=%d seq=%d heads=%d kv_heads=%d scale=%f\n",
            flash_params.causal,
            flash_params.seq_len,
            flash_params.num_heads,
            flash_params.num_kv_heads,
            flash_params.scale
        );
        printf(
            "[ATTENTION] FLASH ATTENTION: "
            "seq=%d heads=32 kv_heads=8 d_head=64\n",
            tokens
        );
        transformer::launch_flash_attention(
            flash_params
        );
    // =========================================================
    // DEBUG: RAW FLASH ATTENTION OUTPUT
    // =========================================================
    cudaDeviceSynchronize();
    __nv_bfloat16 debug[4];

    cudaMemcpy(
        debug,
        flash_output.data_bf16(),
        4 * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );

    printf(
        "CUDA FLASH RAW: %f %f %f %f\n",
        __bfloat162float(debug[0]),
        __bfloat162float(debug[1]),
        __bfloat162float(debug[2]),
        __bfloat162float(debug[3])
    );

    dump_bf16(
        "/tmp/cuda_flash_attention.txt",
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

    dump_bf16(
        "/tmp/cuda_attention_input.txt",
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
    dump_bf16(
        "/tmp/cuda_o_proj.txt",
        o_proj_,
        2048 * 2048
    );
    printf(
        "[ATTENTION] O GEMM: M=%d N=2048 K=2048\n",
        tokens
    );
    launch_gemm_2048x2048(
        attention_input.data_bf16(),
        o_proj_,
        output.data_bf16(),
        tokens,
        2048,
        2048
    );
    cudaDeviceSynchronize();

    printf("O GEMM first values:\n");

    std::vector<__nv_bfloat16> h_output(tokens * 2048);

    cudaMemcpy(
        h_output.data(),
        output.data_bf16(),
        tokens * 2048 * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );

    for(int i=0;i<10;i++)
    {
        printf("%f\n",
            __bfloat162float(h_output[i])
        );
    }

    dump_bf16(
        "/tmp/cuda_attention_output.txt",
        output.data_bf16(),
        tokens * 2048
    );
    }
}