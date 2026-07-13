#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

void gemm_dispatch(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M,
    int N,
    int K
);