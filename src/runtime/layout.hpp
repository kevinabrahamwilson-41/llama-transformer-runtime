#pragma once

#include <cuda_bf16.h>

void launch_flash_output_layout(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int tokens
);