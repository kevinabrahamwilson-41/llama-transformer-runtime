#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "rope.hpp"


// ======================================================
// Llama 3.2 1B RoPE Validation Test
// ======================================================

constexpr int TOKENS = 512;

constexpr int HEADS = ROPE_HEADS;
constexpr int KV_HEADS = ROPE_KV_HEADS;
constexpr int HEAD_DIM = ROPE_HEAD_DIM;
constexpr int ROTARY_DIM = ROPE_ROTARY_DIM;


// ======================================================
// CPU Reference RoPE
// ======================================================

void cpu_rope(
    const std::vector<__nv_bfloat16>& input,
    std::vector<__nv_bfloat16>& output,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    int heads,
    int tokens)
{
    for(int pos = 0; pos < tokens; pos++)
    {
        for(int head = 0; head < heads; head++)
        {
            for(int pair = 0; pair < ROTARY_DIM; pair++)
            {

                int idx =
                    pos * heads * HEAD_DIM +
                    head * HEAD_DIM +
                    pair * 2;


                float x0 =
                    __bfloat162float(input[idx]);

                float x1 =
                    __bfloat162float(input[idx+1]);


                float c =
                    cos_table[pos * ROTARY_DIM + pair];

                float s =
                    sin_table[pos * ROTARY_DIM + pair];


                float y0 =
                    x0 * c - x1 * s;

                float y1 =
                    x0 * s + x1 * c;


                int out_idx =
                    head * tokens * HEAD_DIM +
                    pos * HEAD_DIM +
                    pair * 2;


                output[out_idx] =
                    __float2bfloat16(y0);

                output[out_idx+1] =
                    __float2bfloat16(y1);
            }
        }
    }
}


// ======================================================
// CPU V transpose
// ======================================================

void cpu_transpose_v(
    const std::vector<__nv_bfloat16>& input,
    std::vector<__nv_bfloat16>& output)
{

    for(int pos = 0; pos < TOKENS; pos++)
    {
        for(int head = 0; head < KV_HEADS; head++)
        {
            for(int i = 0; i < HEAD_DIM; i++)
            {

                int src =
                    pos * KV_HEADS * HEAD_DIM +
                    head * HEAD_DIM +
                    i;


                int dst =
                    head * TOKENS * HEAD_DIM +
                    pos * HEAD_DIM +
                    i;


                output[dst] = input[src];
            }
        }
    }
}



// ======================================================
// Compare tensors
// ======================================================

bool validate(
    const std::vector<__nv_bfloat16>& gpu,
    const std::vector<__nv_bfloat16>& cpu,
    const char* name)
{

    float max_error = 0.0f;

    int mismatch = 0;


    for(size_t i = 0; i < gpu.size(); i++)
    {

        float a =
            __bfloat162float(gpu[i]);

        float b =
            __bfloat162float(cpu[i]);


        float diff =
            fabs(a-b);


        max_error =
            std::max(max_error,diff);


        if(diff > 1e-2f)
        {
            mismatch++;

            if(mismatch == 1)
            {
                std::cout
                    << "\nFirst mismatch in "
                    << name
                    << "\n";

                std::cout
                    << "Index    : "
                    << i
                    << "\n";

                std::cout
                    << "GPU      : "
                    << a
                    << "\n";

                std::cout
                    << "CPU      : "
                    << b
                    << "\n";
            }
        }
    }


    std::cout
        << name
        << " Max Error : "
        << max_error
        << "\n";


    std::cout
        << name
        << " Mismatches : "
        << mismatch
        << "\n";


    return mismatch == 0;
}



// ======================================================
// MAIN
// ======================================================

int main()
{

    std::cout
        << "=============================================\n";

    std::cout
        << "        Llama 3.2 CUDA RoPE Validation\n";

    std::cout
        << "=============================================\n\n";


    std::cout
        << "Tokens      : "
        << TOKENS
        << "\n";

    std::cout
        << "Q Heads     : "
        << HEADS
        << "\n";

    std::cout
        << "KV Heads    : "
        << KV_HEADS
        << "\n";

    std::cout
        << "Head Dim    : "
        << HEAD_DIM
        << "\n\n";



    // --------------------------------------------------
    // Allocate host tensors
    // --------------------------------------------------

    std::vector<__nv_bfloat16> h_q(
        TOKENS * HEADS * HEAD_DIM);

    std::vector<__nv_bfloat16> h_k(
        TOKENS * KV_HEADS * HEAD_DIM);

    std::vector<__nv_bfloat16> h_v(
        TOKENS * KV_HEADS * HEAD_DIM);



    std::vector<__nv_bfloat16> q_gpu(
        h_q.size());

    std::vector<__nv_bfloat16> k_gpu(
        h_k.size());

    std::vector<__nv_bfloat16> v_gpu(
        h_v.size());



    std::vector<__nv_bfloat16> q_cpu(
        h_q.size());

    std::vector<__nv_bfloat16> k_cpu(
        h_k.size());

    std::vector<__nv_bfloat16> v_cpu(
        h_v.size());



    // --------------------------------------------------
    // Initialize
    // --------------------------------------------------

    std::mt19937 rng(1234);

    std::uniform_real_distribution<float> dist(-1.0f,1.0f);



    for(auto &x:h_q)
        x = __float2bfloat16(dist(rng));


    for(auto &x:h_k)
        x = __float2bfloat16(dist(rng));


    for(auto &x:h_v)
        x = __float2bfloat16(dist(rng));



    // --------------------------------------------------
    // Generate RoPE tables
    // --------------------------------------------------

    std::vector<float> cos_table(
        TOKENS * ROTARY_DIM);

    std::vector<float> sin_table(
        TOKENS * ROTARY_DIM);



    for(int pos=0; pos<TOKENS; pos++)
    {
        for(int i=0;i<ROTARY_DIM;i++)
        {

            float theta =
                pos /
                powf(
                    10000.0f,
                    (2.0f*i)/HEAD_DIM
                );


            cos_table[
                pos*ROTARY_DIM+i
            ] = cosf(theta);


            sin_table[
                pos*ROTARY_DIM+i
            ] = sinf(theta);
        }
    }



    // --------------------------------------------------
    // Device allocation
    // --------------------------------------------------

    __nv_bfloat16 *dq,*dk,*dv;
    __nv_bfloat16 *dq_out,*dk_out,*dv_out;

    float *d_cos,*d_sin;



    cudaMalloc(&dq,sizeof(__nv_bfloat16)*h_q.size());
    cudaMalloc(&dk,sizeof(__nv_bfloat16)*h_k.size());
    cudaMalloc(&dv,sizeof(__nv_bfloat16)*h_v.size());


    cudaMalloc(&dq_out,sizeof(__nv_bfloat16)*h_q.size());
    cudaMalloc(&dk_out,sizeof(__nv_bfloat16)*h_k.size());
    cudaMalloc(&dv_out,sizeof(__nv_bfloat16)*h_v.size());


    cudaMalloc(&d_cos,sizeof(float)*cos_table.size());
    cudaMalloc(&d_sin,sizeof(float)*sin_table.size());



    cudaMemcpy(dq,h_q.data(),
        sizeof(__nv_bfloat16)*h_q.size(),
        cudaMemcpyHostToDevice);


    cudaMemcpy(dk,h_k.data(),
        sizeof(__nv_bfloat16)*h_k.size(),
        cudaMemcpyHostToDevice);


    cudaMemcpy(dv,h_v.data(),
        sizeof(__nv_bfloat16)*h_v.size(),
        cudaMemcpyHostToDevice);



    cudaMemcpy(d_cos,cos_table.data(),
        sizeof(float)*cos_table.size(),
        cudaMemcpyHostToDevice);


    cudaMemcpy(d_sin,sin_table.data(),
        sizeof(float)*sin_table.size(),
        cudaMemcpyHostToDevice);



    // --------------------------------------------------
    // Launch CUDA RoPE
    // --------------------------------------------------

    std::cout
        << "Launching CUDA RoPE kernel...\n";


    launch_rope_qkv(
        dq,
        dq_out,

        dk,
        dk_out,

        dv,
        dv_out,

        d_cos,
        d_sin,

        TOKENS);



    cudaMemcpy(q_gpu.data(),dq_out,
        sizeof(__nv_bfloat16)*q_gpu.size(),
        cudaMemcpyDeviceToHost);


    cudaMemcpy(k_gpu.data(),dk_out,
        sizeof(__nv_bfloat16)*k_gpu.size(),
        cudaMemcpyDeviceToHost);


    cudaMemcpy(v_gpu.data(),dv_out,
        sizeof(__nv_bfloat16)*v_gpu.size(),
        cudaMemcpyDeviceToHost);



    // --------------------------------------------------
    // CPU reference
    // --------------------------------------------------

    cpu_rope(
        h_q,
        q_cpu,
        cos_table,
        sin_table,
        HEADS,
        TOKENS);


    cpu_rope(
        h_k,
        k_cpu,
        cos_table,
        sin_table,
        KV_HEADS,
        TOKENS);


    cpu_transpose_v(
        h_v,
        v_cpu);



    // --------------------------------------------------
    // Validation
    // --------------------------------------------------

    bool pass = true;


    pass &= validate(
        q_gpu,
        q_cpu,
        "Q RoPE");


    pass &= validate(
        k_gpu,
        k_cpu,
        "K RoPE");


    pass &= validate(
        v_gpu,
        v_cpu,
        "V Transpose");



    std::cout << "\n";


    if(pass)
        std::cout
            << "RoPE Validation : PASSED\n";
    else
        std::cout
            << "RoPE Validation : FAILED\n";



    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);

    cudaFree(dq_out);
    cudaFree(dk_out);
    cudaFree(dv_out);

    cudaFree(d_cos);
    cudaFree(d_sin);


    return 0;
}