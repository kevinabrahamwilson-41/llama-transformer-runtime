#pragma once

#include <cuda_bf16.h>

void launch_silu_mul(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    int seq_len,
    int intermediate = 8192
);