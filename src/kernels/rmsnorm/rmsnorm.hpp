#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

template<int HIDDEN>
void rmsnorm_launch(
    const __nv_bfloat16* d_in,
    const __nv_bfloat16* d_weight,
    __nv_bfloat16* d_out,
    int rows,
    float eps,
    int threads_per_block = 256
);