#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

void launch_gemm_2048x128256(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M,
    int N,
    int K
);