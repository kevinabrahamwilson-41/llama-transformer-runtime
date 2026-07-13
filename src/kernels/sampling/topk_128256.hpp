#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>


constexpr int LLAMA_VOCAB_SIZE = 128256;


void launch_topk(
    const __nv_bfloat16* logits,
    float* values,
    int* indices,
    int k,
    cudaStream_t stream = 0
);