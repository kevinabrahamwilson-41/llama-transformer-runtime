#include "residual_add.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>

namespace transformer {
constexpr int BLOCK_SIZE = 128;

__global__ void residual_add_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int64_t num_elements
)
{
    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_elements)
    {
        float av =
            __bfloat162float(a[idx]);

        float bv =
            __bfloat162float(b[idx]);

        float sum =
            av + bv;

        out[idx] =
            __float2bfloat16_rn(sum);
    }
}

void residual_add(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int64_t num_elements
)
{
    if (a == nullptr || b == nullptr || out == nullptr) {
        throw std::invalid_argument(
            "residual_add: null device pointer"
        );
    }

    if (num_elements <= 0) {
        throw std::invalid_argument(
            "residual_add: num_elements must be positive"
        );
    }

    const int blocks = static_cast<int>(
        (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    residual_add_kernel<<<blocks, BLOCK_SIZE>>>(
        a,
        b,
        out,
        num_elements
    );

    const cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            "residual_add kernel launch failed: " +
            std::string(cudaGetErrorString(error))
        );
    }
}
}  // namespace transformer