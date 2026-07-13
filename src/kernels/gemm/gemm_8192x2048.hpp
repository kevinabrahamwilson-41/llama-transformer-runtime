#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

void launch_gemm_8192x2048(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M,
    int N,
    int K
);