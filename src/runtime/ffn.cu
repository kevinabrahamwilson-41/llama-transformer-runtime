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
namespace runtime
{
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
static void dump_tensor(
    const char* path,
    const __nv_bfloat16* device_tensor,
    int elements
)
{
    std::vector<__nv_bfloat16> host(elements);

    cudaMemcpy(
        host.data(),
        device_tensor,
        elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );


    FILE* file = std::fopen(path, "w");

    if (!file)
    {
        printf("Failed to open dump file\n");
        exit(EXIT_FAILURE);
    }


    for(int i = 0; i < elements; i++)
    {
        float value =
            __bfloat162float(host[i]);

        fprintf(
            file,
            "%.9g\n",
            value
        );
    }


    fclose(file);

    printf(
        "[TEST] Dumped tensor: %s\n",
        path
    );
}
void FeedForward::forward(
    const Tensor& input,
    Tensor& output
) const
{   int tokens = input.shape()[0];
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
    dump_tensor(
        "/tmp/cuda_ffn_norm.txt",
        normalized.data_bf16(),
        tokens * 2048
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
    dump_tensor(
        "/tmp/cuda_gate.txt",
        gate.data_bf16(),
        tokens * 8192
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
    dump_tensor(
        "/tmp/cuda_up.txt",
        up.data_bf16(),
        tokens * 8192
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
    dump_tensor(
        "/tmp/cuda_activated.txt",
        activated.data_bf16(),
        tokens * 8192
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
    dump_tensor(
        "/tmp/cuda_down_proj_output.txt",
        output.data_bf16(),
        tokens * 2048
    );
}

} // namespace runtime