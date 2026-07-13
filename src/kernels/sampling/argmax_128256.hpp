#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>


void launch_argmax(
    const __nv_bfloat16* logits,
    int* token,
    cudaStream_t stream = 0
);