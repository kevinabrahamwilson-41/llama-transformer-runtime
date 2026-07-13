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
// ================= TEST =================
int main()
{
    printf("Testing Llama 3.2 1B TopK Kernel\n");
    int K=10;
    std::vector<__nv_bfloat16> h_logits(
        LLAMA_VOCAB_SIZE
    );
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(
        -10,
        10
    );
    std::vector<std::pair<float,int>> cpu;
    for(int i=0;i<LLAMA_VOCAB_SIZE;i++)
    {
        float v=dist(rng);
        if(i<10)
            v=100-i;
        h_logits[i]=__float2bfloat16(v);
        // simulate GPU BF16 conversion
        float bf =
            __bfloat162float(h_logits[i]);
        cpu.push_back({bf,i});
    }
    std::sort(
        cpu.begin(),
        cpu.end(),
        [](const auto& a, const auto& b)
        {
            constexpr float EPS = 1e-5f;
            if(a.first > b.first + EPS)
                return true;
            if(std::abs(a.first - b.first) <= EPS)
                return a.second < b.second;
            return false;
        }
    );
    __nv_bfloat16* d_logits;
    float* d_values;
    int* d_indices;
    cudaMalloc(
        &d_logits,
        LLAMA_VOCAB_SIZE*sizeof(__nv_bfloat16)
    );
    cudaMalloc(
        &d_values,
        K*sizeof(float)
    );
    cudaMalloc(
        &d_indices,
        K*sizeof(int)
    );
    cudaMemcpy(
        d_logits,
        h_logits.data(),
        LLAMA_VOCAB_SIZE*sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice
    );
    launch_topk(
        d_logits,
        d_values,
        d_indices,
        K
    );
    cudaDeviceSynchronize();
    std::vector<float> values(K);
    std::vector<int> indices(K);
    cudaMemcpy(
        values.data(),
        d_values,
        K*sizeof(float),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        indices.data(),
        d_indices,
        K*sizeof(int),
        cudaMemcpyDeviceToHost
    );
    printf("\nTop K results:\n");
    for(int i=0;i<K;i++)
    {
        printf(
            "%d : %.2f\n",
            indices[i],
            values[i]
        );
    }
    bool pass=true;
    for(int i=0;i<K;i++)
    {
        if(indices[i]!=cpu[i].second)
            pass=false;
    }
    printf(
        pass ?
        "PASS\n":
        "FAIL\n"
    );
    cudaFree(d_logits);
    cudaFree(d_values);
    cudaFree(d_indices);
    return 0;
}