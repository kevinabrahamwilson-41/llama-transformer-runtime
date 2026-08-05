#include "argmax_128256.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cfloat>

constexpr int VOCAB_SIZE = 128256;
constexpr int BLOCK_SIZE = 256;
struct Pair {
    float value;
    int index;
};
__device__ __forceinline__
Pair better(Pair a, Pair b)
{
    if (b.value > a.value)
        return b;

    return a;
}
__device__ __forceinline__
Pair warp_reduce_max(Pair x)
{
    unsigned mask = 0xffffffff;
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        Pair y;
        y.value =
            __shfl_down_sync(mask, x.value, offset);
        y.index =
            __shfl_down_sync(mask, x.index, offset);
        x = better(x,y);
    }
    return x;
}

__global__
void argmax_bf16_128256(
        const __nv_bfloat16* __restrict__ logits,
        int* output)
{
    __shared__ Pair shared[32];
    int tid = threadIdx.x;
    Pair local;
    local.value = -FLT_MAX;
    local.index = -1;
    // grid stride
    for(int i = tid;
        i < VOCAB_SIZE;
        i += BLOCK_SIZE)
    {
        float v =
            __bfloat162float(logits[i]);
        if(v > local.value)
        {
            local.value = v;
            local.index = i;
        }
    }
    // warp reduction
    local = warp_reduce_max(local);
    // one result per warp
    if((tid & 31)==0)
    {
        shared[tid/32] = local;
    }
    __syncthreads();
    // final reduction by warp 0

    if(tid < 32)
    {
        Pair x;
        if(tid < BLOCK_SIZE/32)
            x = shared[tid];
        else
        {
            x.value = -FLT_MAX;
            x.index = -1;
        }
        x = warp_reduce_max(x);
        if(tid==0)
        {
            *output = x.index;
        }
    }
}

void launch_argmax(
    const __nv_bfloat16* logits,
    int* token,
    cudaStream_t stream)
{
    argmax_bf16_128256<<<1,BLOCK_SIZE,0,stream>>>(
        logits,
        token
    );
}