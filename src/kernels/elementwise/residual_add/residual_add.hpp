#pragma once

#include <cuda_bf16.h>

namespace transformer {

void residual_add(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int64_t num_elements
);

} // namespace transformer