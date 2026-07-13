#include "flashattention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
using namespace transformer;
int main() {
    constexpr int BATCH      = 1;
    constexpr int HEADS      = 32;
    constexpr int KV_HEADS   = 8;
    constexpr int SEQ_LEN    = 512;
    constexpr int HEAD_DIM   = 64;
    const size_t q_elements =
        (size_t)BATCH * HEADS * SEQ_LEN * HEAD_DIM;
    const size_t kv_elements =
        (size_t)BATCH * KV_HEADS * SEQ_LEN * HEAD_DIM;
    std::vector<half> hQ(q_elements);
    std::vector<half> hK(kv_elements);
    std::vector<half> hV(kv_elements);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(size_t i=0;i<q_elements;i++)
        hQ[i]=__float2half(dist(rng));
    for(size_t i=0;i<kv_elements;i++){
        hK[i]=__float2half(dist(rng));
        hV[i]=__float2half(dist(rng));
    }
    half *dQ,*dK,*dV,*dO;
    float *dL;
    cudaMalloc(&dQ,q_elements*sizeof(half));
    cudaMalloc(&dK,kv_elements*sizeof(half));
    cudaMalloc(&dV,kv_elements*sizeof(half));
    cudaMalloc(&dO,q_elements*sizeof(half));
    cudaMalloc(&dL,BATCH*HEADS*SEQ_LEN*sizeof(float));
    cudaMemcpy(dQ,hQ.data(),q_elements*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(dK,hK.data(),kv_elements*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(dV,hV.data(),kv_elements*sizeof(half),cudaMemcpyHostToDevice);
    FlashAttentionParams p{};
    p.Q = dQ;
    p.K = dK;
    p.V = dV;
    p.O = dO;
    p.L = dL;
    p.batch_size  = BATCH;
    p.num_heads   = HEADS;
    p.num_kv_heads= KV_HEADS;
    p.seq_len = SEQ_LEN;
    p.d_head  = HEAD_DIM;
    p.scale = 1.f/std::sqrt((float)HEAD_DIM);
    p.causal = true;
    p.dtype = DType::FP16;
    p.autotune = false;
    p.stream = 0;
    launch_flash_attention(p);
    cudaDeviceSynchronize();
    std::vector<half> hO(q_elements);
    cudaMemcpy(
        hO.data(),
        dO,
        q_elements*sizeof(half),
        cudaMemcpyDeviceToHost);
    std::cout<<"FlashAttention finished.\n";
    std::cout<<"First 10 outputs:\n";
    for(int i=0;i<10;i++)
        std::cout<<__half2float(hO[i])<<" ";
    std::cout<<"\n";
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    cudaFree(dL);
    return 0;
}
/*
 2042  nvcc -arch=sm_89 -O3 -c fa_api.cu
 2043  nvcc -arch=sm_89 -O3 -c fa_autotune.cu
 2044  nvcc -arch=sm_89 -O3 -c flashattention.cu
 2045  nvcc -arch=sm_89 -O3 -c flashattention_decode.cu
 2046  ls *.o
 2047  nvcc -arch=sm_89 -O3 flash_test.cu fa_api.o fa_autotune.o flashattention.o flashattention_decode.o -o flash_test
 2048  ./flash_test
 2049  ncu --set full --kernel-name-base demangled ./flash_test
*/