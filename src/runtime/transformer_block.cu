#include "transformer_block.hpp"
#include "../kernels/residual_add/residual_add.hpp"
#include <iostream>
#include <cstdint>
#include <vector>
#include <utility>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace runtime
{

static void debug_check_tensor(
    const char* name,
    const Tensor& tensor,
    int tokens
)
{
    const int elements = tokens * 2048;

    std::vector<__nv_bfloat16> host(
        elements
    );

    cudaMemcpy(
        host.data(),
        tensor.data_bf16(),
        elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );
/*
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
    }*/
/*
    printf(
        "%s max_abs = %.6f at index %d\n",
        name,
        max_abs,
        max_idx
    );*/
}

static void dump_tensor(
    const char* path,
    const Tensor& tensor,
    int tokens
)
{
    const int elements = tokens * 2048;

    std::vector<__nv_bfloat16> host(elements);

    cudaMemcpy(
        host.data(),
        tensor.data_bf16(),
        elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );

    FILE* file = fopen(path, "w");

    if (!file)
    {
        printf("Failed to open dump file: %s\n", path);
        return;
    }

    for (int i = 0; i < elements; ++i)
    {
        fprintf(
            file,
            "%.9g\n",
            __bfloat162float(host[i])
        );
    }

    fclose(file);
}

static float read_device_bf16(
    const __nv_bfloat16* device_ptr
)
{
    __nv_bfloat16 host_value;

    cudaMemcpy(
        &host_value,
        device_ptr,
        sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );

    return __bfloat162float(host_value);
}
TransformerBlock::TransformerBlock(
    Attention&& attention,
    FeedForward&& ffn
)
    : attention_(std::move(attention)),
      ffn_(std::move(ffn))
{
}

void TransformerBlock::forward(
    const Tensor& input,
    Tensor& output,
    int position 
) 
{
    const int tokens =
        static_cast<int>(
            input.shape()[0]
        );

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
        position
    );
    cudaDeviceSynchronize();

    dump_tensor(
        "/tmp/cuda_attention_output.txt",
        attention_output,
        tokens
    );
    debug_check_tensor(
        "ATTENTION",
        attention_output,
        tokens
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
    dump_tensor(
        "/tmp/cuda_residual.txt",
        residual,
        tokens
    );
    debug_check_tensor(
        "RESIDUAL",
        residual,
        tokens
    );
    //std::cout << "RESIDUAL[0] = "
      //      << read_device_bf16(
        //         residual.data_bf16()
          //      )
            //<< "\n";
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
        ffn_output
    );

    cudaDeviceSynchronize();

    dump_tensor(
        "/tmp/cuda_ffn_output.txt",
        ffn_output,
        tokens
    );
    debug_check_tensor(
        "FFN",
        ffn_output,
        tokens
    );
    cudaDeviceSynchronize();
    //std::cout << "FFN[0] = "
      //      << read_device_bf16(
        //         ffn_output.data_bf16()
          //      )
            //<< "\n";
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

    dump_tensor(
        "/tmp/cuda_block0_output.txt",
        output,
        tokens
    );

    debug_check_tensor(
        "OUTPUT",
        output,
        tokens
    );
    //cudaDeviceSynchronize();
    //std::cout << "OUTPUT[0] = "
      //      << read_device_bf16(
        //             output.data_bf16()
          //          )
            //<< "\n";
}

} // namespace runtime