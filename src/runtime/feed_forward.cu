#include "tensor.hpp"
#include "weights.hpp"
#include "feed_forward.hpp"
#include "../kernels/gemm/gemm_2048x8192.hpp"
#include "../kernels/gemm/gemm_8192x2048.hpp"
#include "../kernels/silu/silu_8192.hpp"
#include "feed_forward.hpp"
#include <cuda_runtime.h>
namespace runtime
{
FeedForward::FeedForward(
    __nv_bfloat16* gate_proj,
    __nv_bfloat16* up_proj,
    __nv_bfloat16* down_proj
)
    : gate_proj_(gate_proj),
      up_proj_(up_proj),
      down_proj_(down_proj)
{
}

void FeedForward::forward(
    const Tensor& input,
    Tensor& output
) const
{
    // =========================================================
    // Temporary tensors
    // =========================================================

    Tensor gate(
        {1, 8192},
        DataType::BF16
    );

    Tensor up(
        {1, 8192},
        DataType::BF16
    );

    Tensor activated(
        {1, 8192},
        DataType::BF16
    );

    // =========================================================
    // W1 / gate projection
    //
    // [1, 2048] × [2048, 8192]
    //              ↓
    //          [1, 8192]
    // =========================================================

    launch_gemm_2048x8192(
        input.data_bf16(),
        gate_proj_,
        gate.data_bf16(),
        1,
        8192,
        2048
    );

    // =========================================================
    // W3 / up projection
    //
    // [1, 2048] × [2048, 8192]
    //              ↓
    //          [1, 8192]
    // =========================================================

    launch_gemm_2048x8192(
        input.data_bf16(),
        up_proj_,
        up.data_bf16(),
        1,
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
        8192,
        0
    );

    // =========================================================
    // W2 / down projection
    //
    // [1, 8192] × [8192, 2048]
    //              ↓
    //          [1, 2048]
    // =========================================================

    launch_gemm_8192x2048(
        activated.data_bf16(),
        down_proj_,
        output.data_bf16(),
        1,
        2048,
        8192
    );
}

} // namespace runtime