#include "tensor.hpp"
#include "weights.hpp"
#include "ffn.hpp"
#include "../kernels/gemm/gemm_2048x8192.hpp"
#include "../kernels/gemm/gemm_8192x2048.hpp"
#include "../kernels/silu/silu_8192.hpp"
#include "../kernels/rmsnorm/rmsnorm.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <atomic>
namespace runtime
{
// Accumulator for FFN forward GPU time (ms)
double ffn_accumulated_ms = 0.0;
void FeedForward::reset_ffn_timing(){
    ffn_accumulated_ms = 0.0;
}
double FeedForward::get_accumulated_ffn_ms(){
    return ffn_accumulated_ms;
}
FeedForward::FeedForward(
    __nv_bfloat16* ffn_norm,
    __nv_bfloat16* gate_proj,
    __nv_bfloat16* up_proj,
    __nv_bfloat16* down_proj
)
    : ffn_norm_(ffn_norm),
      gate_proj_(gate_proj),
      up_proj_(up_proj),
      down_proj_(down_proj)
{
}
void FeedForward::forward(
    const Tensor& input,
    Tensor& output,
    int seq_len
) const
{   int tokens = seq_len;
    // start FFN timer (GPU)
    cudaEvent_t _ffn_start_evt, _ffn_stop_evt;
    cudaEventCreate(&_ffn_start_evt);
    cudaEventCreate(&_ffn_stop_evt);
    cudaEventRecord(_ffn_start_evt);
    // =========================================================
    // Temporary tensors
    // =========================================================
    Tensor normalized(
        {tokens, 2048},
        DataType::BF16
    );

    Tensor gate(
        {tokens, 8192},
        DataType::BF16
    );

    Tensor up(
        {tokens, 8192},
        DataType::BF16
    );

    Tensor activated(
        {tokens, 8192},
        DataType::BF16
    );
    // =========================================================
    // RMSNorm
    // =========================================================
    rmsnorm_launch<2048>(
        input.data_bf16(),
        ffn_norm_,
        normalized.data_bf16(),
        tokens,
        1e-5f
    );
    // =========================================================
    // W1 / gate projection
    //
    // [tokens, 2048] × [2048, 8192]
    //                    ↓
    //                [tokens, 8192]
    // =========================================================

    launch_gemm_2048x8192(
        normalized.data_bf16(),
        gate_proj_,
        gate.data_bf16(),
        tokens,
        8192,
        2048
    );

    // =========================================================
    // W3 / up projection
    //
    // [tokens, 2048] × [2048, 8192]
    //                    ↓
    //                [tokens, 8192]
    // =========================================================

    launch_gemm_2048x8192(
        normalized.data_bf16(),
        up_proj_,
        up.data_bf16(),
        tokens,
        8192,
        2048
    );
    // =========================================================
    // SiLU(gate) * up
    // =========================================================

    launch_silu_mul(
        gate.data_bf16(),
        up.data_bf16(),
        activated.data_bf16(),
        tokens,
        8192
    );
    // =========================================================
    // W2 / down projection
    //
    // [tokens, 8192] × [8192, 2048]
    //                    ↓
    //                [tokens, 2048]
    // =========================================================

    launch_gemm_8192x2048(
        activated.data_bf16(),
        down_proj_,
        output.data_bf16(),
        tokens,
        2048,
        8192
    );
    // stop FFN timer and accumulate
    cudaEventRecord(_ffn_stop_evt);
    cudaEventSynchronize(_ffn_stop_evt);
    float _ffn_elapsed_ms = 0.0f;
    cudaEventElapsedTime(&_ffn_elapsed_ms, _ffn_start_evt, _ffn_stop_evt);
    ffn_accumulated_ms += static_cast<double>(_ffn_elapsed_ms);
    cudaEventDestroy(_ffn_start_evt);
    cudaEventDestroy(_ffn_stop_evt);
}

} // namespace runtime