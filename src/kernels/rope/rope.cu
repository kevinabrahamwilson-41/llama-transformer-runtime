#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <rope.hpp>
// ======================================================
// Llama 3.2 1B Constants
// ======================================================
constexpr int SEQ_LEN  = ROPE_SEQ_LEN;
constexpr int HEADS    = ROPE_HEADS;
constexpr int KV_HEADS = ROPE_KV_HEADS;
constexpr int HEAD_DIM = ROPE_HEAD_DIM;
constexpr int BLOCK_SIZE = ROPE_BLOCK_SIZE;
// RoPE dimensions
constexpr int ROTARY_DIM = ROPE_ROTARY_DIM;
// ======================================================
// CUDA Error Checking
// ======================================================
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if(err != cudaSuccess)                                    \
    {                                                         \
        printf("CUDA Error %s:%d : %s\n",                     \
               __FILE__,                                     \
               __LINE__,                                     \
               cudaGetErrorString(err));                      \
        exit(EXIT_FAILURE);                                  \
    }                                                         \
} while(0)
// ======================================================
// RoPE rotation
//
// [x0,x1] -> [x0*cos - x1*sin,
//             x0*sin + x1*cos]
//
// ======================================================
__global__
void rope_q_kernel(
    __nv_bfloat16* q,
    const float* cos_table,
    const float* sin_table
){
    int warp_id = blockIdx.x * (blockDim.x / 32)
                + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int total_warps = SEQ_LEN * HEADS;
    if(warp_id >= total_warps)
        return;
    // one warp = one token position + one head
    int pos = warp_id / HEADS;
    int head = warp_id % HEADS;
    // lane handles one pair
    int pair = lane;
    int offset =
        pos * HEADS * HEAD_DIM
        +
        head * HEAD_DIM
        +
        pair * 2;
    float c =
        cos_table[
            pos * ROTARY_DIM + pair
        ];
    float s =
        sin_table[
            pos * ROTARY_DIM + pair
        ];
    __nv_bfloat162 x =
        *reinterpret_cast<__nv_bfloat162*>(&q[offset]);
    float x0 =
        __bfloat162float(x.x);
    float x1 =
        __bfloat162float(x.y);
    float y0 =
        x0*c - x1*s;
    float y1 =
        x0*s + x1*c;
    __nv_bfloat162 out;
    out.x =
        __float2bfloat16(y0);
    out.y =
        __float2bfloat16(y1);
    *reinterpret_cast<__nv_bfloat162*>(&q[offset])
        = out;
}
__global__
void rope_k_kernel(
    __nv_bfloat16* k,
    const float* cos_table,
    const float* sin_table
){
    int warp_id =
        blockIdx.x * (blockDim.x/32)
        +
        threadIdx.x/32;
    int lane =
        threadIdx.x % 32;
    int total_warps =
        SEQ_LEN * KV_HEADS;
    if(warp_id >= total_warps)
        return;
    int pos =
        warp_id / KV_HEADS;
    int head =
        warp_id % KV_HEADS;
    int pair = lane;
    int offset =
        pos * KV_HEADS * HEAD_DIM
        +
        head * HEAD_DIM
        +
        pair * 2;
    float c =
        cos_table[pos*ROTARY_DIM+pair];
    float s =
        sin_table[pos*ROTARY_DIM+pair];
    __nv_bfloat162 x =
        *reinterpret_cast<__nv_bfloat162*>(&k[offset]);
    float x0 =
        __bfloat162float(x.x);
    float x1 =
        __bfloat162float(x.y);
    float y0 =
        x0*c - x1*s;

    float y1 =
        x0*s + x1*c;
    __nv_bfloat162 out;
    out.x =
        __float2bfloat16(y0);
    out.y =
        __float2bfloat16(y1);
    *reinterpret_cast<__nv_bfloat162*>(&k[offset])
        = out;
}
// ======================================================
// Launcher
// ======================================================
void launch_rope_qk(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    float* cos_table,
    float* sin_table
){
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    int q_warps =
        SEQ_LEN * HEADS;
    int q_blocks =
        (q_warps + WARPS_PER_BLOCK - 1)
        /
        WARPS_PER_BLOCK;
    rope_q_kernel<<<
        q_blocks,
        BLOCK_SIZE
    >>>(
        q,
        cos_table,
        sin_table
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int k_warps =
        SEQ_LEN * KV_HEADS;
    int k_blocks =
        (k_warps + WARPS_PER_BLOCK - 1)
        /
        WARPS_PER_BLOCK;
    rope_k_kernel<<<
        k_blocks,
        BLOCK_SIZE
    >>>(
        k,
        cos_table,
        sin_table
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
// ======================================================
// CPU Reference RoPE
// ======================================================
void rope_reference(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const float* cos_table,
    const float* sin_table
){
    // Q
    for(int pos = 0; pos < SEQ_LEN; pos++){
        for(int head = 0; head < HEADS; head++){
            for(int pair = 0; pair < ROTARY_DIM; pair++){
                int offset =
                    pos * HEADS * HEAD_DIM
                    +
                    head * HEAD_DIM
                    +
                    pair * 2;
                float x0 =
                    __bfloat162float(q[offset]);
                float x1 =
                    __bfloat162float(q[offset+1]);
                float c =
                    cos_table[pos*ROTARY_DIM+pair];

                float s =
                    sin_table[pos*ROTARY_DIM+pair];
                float y0 =
                    x0*c - x1*s;
                float y1 =
                    x0*s + x1*c;
                q[offset] =
                    __float2bfloat16(y0);
                q[offset+1] =
                    __float2bfloat16(y1);

            }
        }
    }
    // K
    for(int pos = 0; pos < SEQ_LEN; pos++){
        for(int head = 0; head < KV_HEADS; head++){
            for(int pair = 0; pair < ROTARY_DIM; pair++){
                int offset =
                    pos * KV_HEADS * HEAD_DIM
                    +
                    head * HEAD_DIM
                    +
                    pair * 2;
                float x0 =
                    __bfloat162float(k[offset]);

                float x1 =
                    __bfloat162float(k[offset+1]);
                float c =
                    cos_table[pos*ROTARY_DIM+pair];

                float s =
                    sin_table[pos*ROTARY_DIM+pair];
                float y0 =
                    x0*c - x1*s;
                float y1 =
                    x0*s + x1*c;
                k[offset] =
                    __float2bfloat16(y0);
                k[offset+1] =
                    __float2bfloat16(y1);
            }
        }
    }
}
// ======================================================
// Verification
// ======================================================
void verify(
    __nv_bfloat16* gpu,
    __nv_bfloat16* ref,
    int elements,
    const char* name
){
    float max_error = 0.0f;
    int errors = 0;
    for(int i=0;i<elements;i++){
        float a =
            __bfloat162float(gpu[i]);
        float b =
            __bfloat162float(ref[i]);
        float err =
            fabsf(a-b);
        max_error =
            fmaxf(max_error,err);
        if(err > 1e-2)
            errors++;
    }
    printf("%s\n",name);
    printf(
        "Max error : %.8f\n",
        max_error
    );
    printf(
        "Errors    : %d / %d\n\n",
        errors,
        elements
    );
}
// ======================================================
// MAIN TEST
// ======================================================
int main(){
    printf(
        "===== Llama 3.2 1B RoPE CUDA Test =====\n"
    );
    constexpr int Q_SIZE =
        SEQ_LEN *
        HEADS *
        HEAD_DIM;
    constexpr int K_SIZE =
        SEQ_LEN *
        KV_HEADS *
        HEAD_DIM;
    size_t q_bytes =
        Q_SIZE*sizeof(__nv_bfloat16);
    size_t k_bytes =
        K_SIZE*sizeof(__nv_bfloat16);
    constexpr int FREQ_SIZE =
        SEQ_LEN *
        ROTARY_DIM;
    // -----------------------------
    // Host allocation
    // -----------------------------
    __nv_bfloat16* h_q =
        new __nv_bfloat16[Q_SIZE];
    __nv_bfloat16* h_k =
        new __nv_bfloat16[K_SIZE];
    __nv_bfloat16* h_q_ref =
        new __nv_bfloat16[Q_SIZE];
    __nv_bfloat16* h_k_ref =
        new __nv_bfloat16[K_SIZE];
    float* h_cos =
        new float[FREQ_SIZE];
    float* h_sin =
        new float[FREQ_SIZE];
    // -----------------------------
    // Initialize data
    // -----------------------------
    srand(1234);
    for(int i=0;i<Q_SIZE;i++){
        float x =
            ((float)rand()/RAND_MAX)
            *2.0f-1.0f;
        h_q[i]=
            __float2bfloat16(x);
        h_q_ref[i]=
            h_q[i];
    }
    for(int i=0;i<K_SIZE;i++){
        float x =
            ((float)rand()/RAND_MAX)
            *2.0f-1.0f;
        h_k[i]=
            __float2bfloat16(x);
        h_k_ref[i]=
            h_k[i];
    }
    for(int pos=0;pos<SEQ_LEN;pos++){
        for(int i=0;i<ROTARY_DIM;i++){
            float theta =
                pos /
                powf(
                    10000.0f,
                    (2.0f*i)/HEAD_DIM
                );
            h_cos[pos*ROTARY_DIM+i]
                = cosf(theta);
            h_sin[pos*ROTARY_DIM+i]
                = sinf(theta);
        }
    }
    // -----------------------------
    // CPU reference
    // -----------------------------
    rope_reference(
        h_q_ref,
        h_k_ref,
        h_cos,
        h_sin
    );
    // -----------------------------
    // Device
    // -----------------------------
    __nv_bfloat16 *d_q;
    __nv_bfloat16 *d_k;
    float *d_cos;
    float *d_sin;
    cudaMalloc(&d_q,q_bytes);
    cudaMalloc(&d_k,k_bytes);
    cudaMalloc(
        &d_cos,
        FREQ_SIZE*sizeof(float)
    );
    cudaMalloc(
        &d_sin,
        FREQ_SIZE*sizeof(float)
    );
    cudaMemcpy(
        d_q,
        h_q,
        q_bytes,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_k,
        h_k,
        k_bytes,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_cos,
        h_cos,
        FREQ_SIZE*sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_sin,
        h_sin,
        FREQ_SIZE*sizeof(float),
        cudaMemcpyHostToDevice
    );
    // -----------------------------
    // CUDA kernel
    // -----------------------------
    launch_rope_qk(
        d_q,
        d_k,
        d_cos,
        d_sin
    );
    cudaDeviceSynchronize();
    cudaMemcpy(
        h_q,
        d_q,
        q_bytes,
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        h_k,
        d_k,
        k_bytes,
        cudaMemcpyDeviceToHost
    );
    // -----------------------------
    // Compare
    // -----------------------------
    verify(
        h_q,
        h_q_ref,
        Q_SIZE,
        "QUERY"
    );
    verify(
        h_k,
        h_k_ref,
        K_SIZE,
        "KEY"
    );
    printf("First 8 Q values after RoPE:\n");
    for(int i=0;i<8;i++){
        printf(
            "%f\n",
            __bfloat162float(h_q[i])
        );
    }
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_cos);
    cudaFree(d_sin);
    delete[] h_q;
    delete[] h_k;
    delete[] h_q_ref;
    delete[] h_k_ref;
    delete[] h_cos;
    delete[] h_sin;
    return 0;
}