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

} // namespace transformer
int main() {
    constexpr int64_t HIDDEN = 2048;

    std::vector<__nv_bfloat16> h_a(HIDDEN);
    std::vector<__nv_bfloat16> h_b(HIDDEN);
    std::vector<__nv_bfloat16> h_out(HIDDEN);

    for (int64_t i = 0; i < HIDDEN; ++i) {
        h_a[i] = __float2bfloat16(1.0f);
        h_b[i] = __float2bfloat16(2.0f);
    }

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_out = nullptr;

    cudaMalloc(
        &d_a,
        HIDDEN * sizeof(__nv_bfloat16)
    );

    cudaMalloc(
        &d_b,
        HIDDEN * sizeof(__nv_bfloat16)
    );

    cudaMalloc(
        &d_out,
        HIDDEN * sizeof(__nv_bfloat16)
    );

    cudaMemcpy(
        d_a,
        h_a.data(),
        HIDDEN * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b,
        h_b.data(),
        HIDDEN * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice
    );

    transformer::residual_add(
        d_a,
        d_b,
        d_out,
        HIDDEN
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_out.data(),
        d_out,
        HIDDEN * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );

    for (int64_t i = 0; i < HIDDEN; ++i) {
        const float result = __bfloat162float(h_out[i]);

        if (std::fabs(result - 3.0f) > 1e-3f) {
            std::cout << "FAIL\n";
            return 1;
        }
    }

    std::cout << "PASS\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}