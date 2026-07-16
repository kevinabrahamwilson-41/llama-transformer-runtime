#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>


void launch_silu_mul(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    int elements,
    cudaStream_t stream
);