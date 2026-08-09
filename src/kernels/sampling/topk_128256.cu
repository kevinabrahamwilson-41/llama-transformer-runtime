#include "topk_128256.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cfloat>
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_K = 64;
struct TopKPair
{
    float value;
    int index;
};
__device__ __forceinline__
bool greater_pair(
    TopKPair a,
    TopKPair b)
{
    return a.value > b.value;
}
__device__ __forceinline__
bool better(
    TopKPair a,
    TopKPair b)
{
    constexpr float EPS = 1e-5f;
    if(a.value > b.value + EPS)
        return true;
    if(fabsf(a.value-b.value) <= EPS &&
       a.index < b.index)
        return true;
    return false;
}

__device__ __forceinline__
void insert_topk(
    TopKPair* list,
    TopKPair item,
    int k)
{

    if(!better(item,list[k-1]))
        return;


    int pos=k-1;


    while(pos>0 &&
          better(item,list[pos-1]))
    {
        list[pos]=list[pos-1];
        pos--;
    }


    list[pos]=item;
}
__global__
void topk_bf16_128256(
        const __nv_bfloat16* __restrict__ logits,
        float* out_values,
        int* out_indices,
        int k)
{
    __shared__ TopKPair shared[8][MAX_K];
    int tid = threadIdx.x;
    TopKPair local[MAX_K];
    for(int i=0;i<k;i++)
    {
        local[i].value = -FLT_MAX;
        local[i].index = -1;
    }
    // each thread scans vocab
    for(int i=tid;
        i<LLAMA_VOCAB_SIZE;
        i+=BLOCK_SIZE)
    {
        TopKPair x;
        x.value =
            __bfloat162float(logits[i]);
        x.index=i;
        insert_topk(
            local,
            x,
            k
        );
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    // only warp leader writes
    if(lane == 0)
    {
        for(int i=0;i<k;i++)
        {
            shared[warp][i]=local[i];
        }
    }
    __syncthreads();
    // thread 0 merge all candidates
    if(tid==0)
    {
        TopKPair final[MAX_K];
        for(int i=0;i<k;i++)
        {
            final[i].value=-FLT_MAX;
            final[i].index=-1;
        }
        for(int t=0;t<8;t++)
        {
            for(int j=0;j<k;j++)
            {
                insert_topk(
                    final,
                    shared[t][j],
                    k
                );
            }
        }
        for(int i=0;i<k;i++)
        {
            out_values[i]=final[i].value;
            out_indices[i]=final[i].index;
        }
    }
}
void launch_topk(
    const __nv_bfloat16* logits,
    float* values,
    int* indices,
    int k,
    cudaStream_t stream)
{
    topk_bf16_128256<<<1,BLOCK_SIZE,0,stream>>>(
        logits,
        values,
        indices,
        k
    );

}