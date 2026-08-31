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
#include <atomic>
namespace runtime{

// Accumulator for attention forward GPU time (ms)
static std::atomic<double> attention_accumulated_ms{0.0};

void Attention::reset_attention_timing(){
    attention_accumulated_ms.store(0.0);
}

double Attention::get_accumulated_attention_ms(){
    return attention_accumulated_ms.load();
}


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
    float* sin_table,
    int max_seq_len,
    int layer_idx
)
    : input_norm_(input_norm),
      q_proj_(q_proj),
      k_proj_(k_proj),
      v_proj_(v_proj),
      o_proj_(o_proj),
      cos_table_(cos_table),
      sin_table_(sin_table),
      max_seq_len_(max_seq_len),
      layer_idx_(layer_idx){
    key_cache_.allocate(
        {8,max_seq_len,64},
        DataType::BF16
    );
    value_cache_.allocate(
        {8,max_seq_len,64},
        DataType::BF16
    );
    cudaMemset(
        key_cache_.data_bf16(),
        0,
        8 * max_seq_len * 64 * sizeof(__nv_bfloat16)
    );
    cudaMemset(
        value_cache_.data_bf16(),
        0,
        8 * max_seq_len * 64 * sizeof(__nv_bfloat16)
    );
}
const __nv_bfloat16* Attention::get_key_cache() const{
    return key_cache_.data_bf16();
}
const __nv_bfloat16* Attention::get_value_cache() const{
    return value_cache_.data_bf16();
}
void Attention::forward(
    const Tensor& input,
    Tensor& output,
    int position,
    int seq_len
)
{
    int tokens = seq_len;
    // start attention timer (GPU)
    cudaEvent_t _att_start_evt, _att_stop_evt;
    cudaEventCreate(&_att_start_evt);
    cudaEventCreate(&_att_stop_evt);
    cudaEventRecord(_att_start_evt);
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
    launch_rope_qkv(
        q.data_bf16(),
        q_flash.data_bf16(),
        k.data_bf16(),
        k_flash.data_bf16(),
        v.data_bf16(),
        v_flash.data_bf16(),
        cos_table_,
        sin_table_,
        tokens,
        position
    );
    cudaDeviceSynchronize();
    for(int h = 0; h < 8; h++){
        for(int t = 0; t < tokens; t++){
            cudaMemcpy(
                key_cache_.data_bf16()
                + h * max_seq_len_ * 64
                + (position + t) * 64,
                k_flash.data_bf16()
                + h * tokens * 64
                + t * 64,
                64 * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice
            );
            cudaMemcpy(
                value_cache_.data_bf16()
                + h * max_seq_len_ * 64
                + (position + t) * 64,
                v_flash.data_bf16()
                + h * tokens * 64
                + t * 64,
                64 * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice
            );
        }
    }
    cudaDeviceSynchronize();
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
            key_cache_.data_bf16();

        flash_params.V =
            value_cache_.data_bf16();

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
        flash_params.q_seq_len =
            tokens;
        bool prefill = (tokens > 1);
        int kv_len;
        if(prefill){
            kv_len = position + tokens;
        }
        else{
            kv_len = position + 1;
        }
        flash_params.kv_seq_len = kv_len;
        // physical stride between KV heads (in tokens) — key/value caches are allocated with max_seq_len spacing
        flash_params.kv_stride = max_seq_len_;
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
    // =========================================================
    // DEBUG: Q AFTER LAYOUT CONVERSION
    // BEFORE FLASH ATTENTION
    // =========================================================
    cudaDeviceSynchronize();
    transformer::launch_flash_attention(flash_params);
    // =========================================================
    // DEBUG: RAW FLASH ATTENTION OUTPUT
    // =========================================================
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
    launch_flash_output_layout(
        flash_output.data_bf16(),
        attention_input.data_bf16(),
        tokens
    );
    cudaDeviceSynchronize();
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
    // stop attention timer and accumulate
    cudaEventRecord(_att_stop_evt);
    cudaEventSynchronize(_att_stop_evt);
    float _att_elapsed_ms = 0.0f;
    cudaEventElapsedTime(&_att_elapsed_ms, _att_start_evt, _att_stop_evt);
    double old_value = attention_accumulated_ms.load(std::memory_order_relaxed);
    while (!attention_accumulated_ms.compare_exchange_weak(
        old_value,
        old_value + static_cast<double>(_att_elapsed_ms),
        std::memory_order_relaxed,
        std::memory_order_relaxed)) {
    }
    cudaEventDestroy(_att_start_evt);
    cudaEventDestroy(_att_stop_evt);
    }
}