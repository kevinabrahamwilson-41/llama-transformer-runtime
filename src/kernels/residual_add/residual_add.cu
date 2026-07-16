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
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t num_elements
) {
    const int64_t pair_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const int64_t element_idx = pair_idx * 2;

    if (element_idx < num_elements) {
        const __nv_bfloat162 a2 =
            *reinterpret_cast<const __nv_bfloat162*>(
                &a[element_idx]
            );

        const __nv_bfloat162 b2 =
            *reinterpret_cast<const __nv_bfloat162*>(
                &b[element_idx]
            );

        const __nv_bfloat162 result =
            __hadd2(a2, b2);

        *reinterpret_cast<__nv_bfloat162*>(
            &out[element_idx]
        ) = result;
    }
}

void residual_add(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int64_t num_elements
) {
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
    const int64_t num_pairs = num_elements / 2;
    const int blocks = static_cast<int>(
        (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE
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