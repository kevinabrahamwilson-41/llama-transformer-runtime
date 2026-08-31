// rmsnorm_pure.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cuda_bf16.h>
#include "rmsnorm.hpp"
// Warp-level reduction for sum
template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-wide reduction for sum
template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    extern __shared__ unsigned char _shared_mem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(_shared_mem);
    int lane = threadIdx.x & 31;      // lane within warp
    int wid  = threadIdx.x >> 5;      // warp ID
    // Number of warps in this block
    int numWarps = (blockDim.x + 31) / 32;
    // Step 1: Reduce within each warp
    val = warpReduceSum(val);
    // Step 2: Warp leaders write their sums to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    // Step 3: First warp loads the warp sums
    val = (lane < numWarps) ? shared[lane] : scalar_t(0);
    // Step 4: First warp reduces the warp sums
    if (wid == 0) {
        val = warpReduceSum(val);
        // Step 5: Store the final block sum
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    // Step 6: Every thread reads the final block sum
    return shared[0];
}

// RMSNorm forward kernel: one block per row
template<int HIDDEN>
__global__ void rmsnorm_fwd_kernel(
    const __nv_bfloat16* __restrict__ input,   // [rows, hidden]
    const __nv_bfloat16* __restrict__ weight,  // [hidden]
    __nv_bfloat16* __restrict__ output,        // [rows, hidden]
    int rows,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* row_in  = input  + row * HIDDEN;
    __nv_bfloat16*       row_out = output + row * HIDDEN;
    // Reinterpret the row as __nv_bfloat162 vectors.
    // HIDDEN must be divisible by 2.
    const uint4* row4 = reinterpret_cast<const uint4*>(row_in);
    // Compute sum of squares over hidden dimension
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = threadIdx.x; i < HIDDEN / 8; i += blockDim.x) {
        uint4 data = row4[i];
        const __nv_bfloat162* bf =
            reinterpret_cast<const __nv_bfloat162*>(&data);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 v =
                __bfloat1622float2(bf[j]);
            sum_sq += v.x * v.x;
            sum_sq += v.y * v.y;
        }
    }
    // Block reduction
    sum_sq = blockReduceSum(sum_sq);
    // Compute RMS reciprocal
    float rms_recip = rsqrtf(sum_sq / static_cast<float>(HIDDEN) + eps);
    // Vectorized pointers
    const uint4* weight4 =
        reinterpret_cast<const uint4*>(weight);
    uint4* out4 =
        reinterpret_cast<uint4*>(row_out);
    #pragma unroll
    for (int i = threadIdx.x; i < HIDDEN / 8; i += blockDim.x) {
        uint4 x_data = row4[i];
        uint4 w_data = weight4[i];
        const __nv_bfloat162* x =
            reinterpret_cast<const __nv_bfloat162*>(&x_data);
        const __nv_bfloat162* w =
            reinterpret_cast<const __nv_bfloat162*>(&w_data);
        __nv_bfloat162 result[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 xf = __bfloat1622float2(x[j]);
            float2 wf = __bfloat1622float2(w[j]);
            float2 yf;
            yf.x = xf.x * rms_recip * wf.x;
            yf.y = xf.y * rms_recip * wf.y;
            result[j] = __floats2bfloat162_rn(yf.x, yf.y);
        }
        out4[i] = *reinterpret_cast<uint4*>(result);
    }
    __syncthreads();
}

// Host helper: launch RMSNorm on [rows x hidden]
template<int HIDDEN>
void rmsnorm_launch(
        const __nv_bfloat16* d_in,
        const __nv_bfloat16* d_weight,
        __nv_bfloat16* d_out,
        int rows,
        float eps,
        int threads_per_block
    ) {
    int blocks = rows;
    int threads = threads_per_block;
    size_t shared_mem = ((threads_per_block + 31) / 32) * sizeof(float);
    rmsnorm_fwd_kernel<HIDDEN><<<blocks, threads, shared_mem, 0>>>(
        d_in,
        d_weight,
        d_out,
        rows,
        eps
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
template void rmsnorm_launch<2048>(
    const __nv_bfloat16*,
    const __nv_bfloat16*,
    __nv_bfloat16*,
    int,
    float,
    int
);
/*
run: nvcc -arch=sm_89 -lineinfo -O3 rmsnorm.cu -o rmsnorm
run: ./rmsnorm
run: sudo /usr/local/cuda/bin/ncu --page details --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section SchedulerStats ./rmsnorm
*/