#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

void launch_gemm_2048x512(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K
);