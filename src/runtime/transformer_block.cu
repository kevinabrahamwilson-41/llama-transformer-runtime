#include "transformer_block.hpp"
#include "../kernels/residual_add/residual_add.hpp"
#include <iostream>
#include <cstdint>
#include <vector>
#include <utility>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace runtime{
TransformerBlock::TransformerBlock(
    Attention&& attention,
    FeedForward&& ffn
)    : attention_(std::move(attention)),
      ffn_(std::move(ffn))
{
}
const __nv_bfloat16* TransformerBlock::get_key_cache() const{
    return attention_.get_key_cache();
}
const __nv_bfloat16* TransformerBlock::get_value_cache() const{
    return attention_.get_value_cache();
}
void TransformerBlock::forward(
    const Tensor& input,
    Tensor& output,
    int position,
    int seq_len
) 
{
    const int tokens = seq_len;
    // =========================================================
    // 1. Attention
    //
    // input
    //   ↓
    // Attention RMSNorm
    //   ↓
    // Attention
    //   ↓
    // attention_output
    //
    // [tokens, 2048]
    // =========================================================
    Tensor attention_output(
        {tokens, 2048},
        DataType::BF16
    );
    attention_.forward(
        input,
        attention_output,
        position,
        seq_len
    );
    cudaDeviceSynchronize();
    //std::cout << "ATTENTION[0] = "
      //      << read_device_bf16(
        //            attention_output.data_bf16()
          //      )
            //<< "\n";
    // =========================================================
    // 2. Attention residual
    //
    // input + attention_output
    //        ↓
    //     residual
    //
    // [tokens, 2048]
    // =========================================================

    Tensor residual(
        {tokens, 2048},
        DataType::BF16
    );

    transformer::residual_add(
        input.data_bf16(),
        attention_output.data_bf16(),
        residual.data_bf16(),
        static_cast<int64_t>(tokens) * 2048
    );
    cudaDeviceSynchronize();
    // =========================================================
    // 3. FeedForward
    //
    // residual
    //    ↓
    // FFN RMSNorm
    //    ↓
    // FFN
    //    ↓
    // ffn_output
    //
    // [tokens, 2048]
    // =========================================================

    Tensor ffn_output(
        {tokens, 2048},
        DataType::BF16
    );
    ffn_.forward(
        residual,
        ffn_output,
        seq_len
    );
    cudaDeviceSynchronize();
    // =========================================================
    // 4. FFN residual
    //
    // residual + ffn_output
    //          ↓
    //        output
    //
    // [tokens, 2048]
    // =========================================================
    transformer::residual_add(
        residual.data_bf16(),
        ffn_output.data_bf16(),
        output.data_bf16(),
        static_cast<int64_t>(tokens) * 2048
    );
    cudaDeviceSynchronize();
}

} // namespace runtime