#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#include "flashattention.h"


#define CHECK(call)                                             \
do {                                                            \
    cudaError_t err = call;                                     \
    if(err != cudaSuccess){                                     \
        std::cout<<"CUDA ERROR: "<<cudaGetErrorString(err);     \
        exit(1);                                                \
    }                                                           \
}while(0)



// ============================================================
// Llama 3.2 1B configuration
// ============================================================

constexpr int BATCH = 1;

constexpr int HEADS = 32;
constexpr int KV_HEADS = 8;

constexpr int HEAD_DIM = 64;

constexpr int SEQ_LEN = 2048;



float bf16_to_float(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}



// ============================================================
// Reference CPU attention
// ============================================================

void reference_attention(
    std::vector<__nv_bfloat16>& Q,
    std::vector<__nv_bfloat16>& K,
    std::vector<__nv_bfloat16>& V,
    std::vector<float>& O
)
{

    float scale = 1.0f / sqrtf(HEAD_DIM);


    for(int h=0; h<HEADS; h++)
    {

        int kvh = h / (HEADS/KV_HEADS);


        for(int q=0;q<SEQ_LEN;q++)
        {

            float scores[SEQ_LEN];


            float maxv=-1e30;


            for(int k=0;k<SEQ_LEN;k++)
            {


                if(k>q)
                {
                    scores[k]=-1e30;
                    continue;
                }


                float dot=0;


                for(int d=0;d<HEAD_DIM;d++)
                {

                    float qv =
                    bf16_to_float(
                    Q[h*SEQ_LEN*HEAD_DIM+
                    q*HEAD_DIM+d]);


                    float kv =
                    bf16_to_float(
                    K[kvh*SEQ_LEN*HEAD_DIM+
                    k*HEAD_DIM+d]);


                    dot+=qv*kv;

                }


                scores[k]=dot*scale;


                maxv=fmax(maxv,scores[k]);

            }



            float sum=0;


            for(int k=0;k<SEQ_LEN;k++)
            {
                scores[k]=expf(scores[k]-maxv);
                sum+=scores[k];
            }



            for(int d=0;d<HEAD_DIM;d++)
            {

                float out=0;


                for(int k=0;k<SEQ_LEN;k++)
                {

                    float vv=
                    bf16_to_float(
                    V[kvh*SEQ_LEN*HEAD_DIM+
                    k*HEAD_DIM+d]);


                    out +=
                    scores[k]*vv;

                }


                O[
                h*SEQ_LEN*HEAD_DIM+
                q*HEAD_DIM+d
                ] = out/sum;

            }

        }

    }

}





int main()
{


std::cout
<<"=============================================\n"
<<" FlashAttention BF16 Validation Benchmark\n"
<<" Llama 3.2 1B Configuration\n"
<<"=============================================\n\n";


size_t q_size =
BATCH*HEADS*SEQ_LEN*HEAD_DIM;


size_t kv_size =
BATCH*KV_HEADS*SEQ_LEN*HEAD_DIM;



std::vector<__nv_bfloat16> h_Q(q_size);
std::vector<__nv_bfloat16> h_K(kv_size);
std::vector<__nv_bfloat16> h_V(kv_size);

std::vector<float> reference(
HEADS*SEQ_LEN*HEAD_DIM);



std::mt19937 rng(1234);

std::uniform_real_distribution<float>
dist(-1.0f,1.0f);



for(auto &x:h_Q)
    x=__float2bfloat16(dist(rng));


for(auto &x:h_K)
    x=__float2bfloat16(dist(rng));


for(auto &x:h_V)
    x=__float2bfloat16(dist(rng));





// ==========================================================
// GPU allocation
// ==========================================================


__nv_bfloat16 *d_Q;
__nv_bfloat16 *d_K;
__nv_bfloat16 *d_V;
__nv_bfloat16 *d_O;

float *d_L;



CHECK(cudaMalloc(&d_Q,sizeof(__nv_bfloat16)*q_size));

CHECK(cudaMalloc(&d_K,sizeof(__nv_bfloat16)*kv_size));

CHECK(cudaMalloc(&d_V,sizeof(__nv_bfloat16)*kv_size));

CHECK(cudaMalloc(&d_O,sizeof(__nv_bfloat16)*q_size));

CHECK(cudaMalloc(&d_L,
sizeof(float)*HEADS*SEQ_LEN));



CHECK(cudaMemcpy(
d_Q,h_Q.data(),
sizeof(__nv_bfloat16)*q_size,
cudaMemcpyHostToDevice));


CHECK(cudaMemcpy(
d_K,h_K.data(),
sizeof(__nv_bfloat16)*kv_size,
cudaMemcpyHostToDevice));


CHECK(cudaMemcpy(
d_V,h_V.data(),
sizeof(__nv_bfloat16)*kv_size,
cudaMemcpyHostToDevice));



// ==========================================================
// Launch
// ==========================================================


transformer::FlashAttentionParams params{};


params.Q=d_Q;
params.K=d_K;
params.V=d_V;
params.O=d_O;
params.L=d_L;


params.seq_len=SEQ_LEN;

params.batch_size=BATCH;

params.num_heads=HEADS;

params.num_kv_heads=KV_HEADS;

params.d_head=HEAD_DIM;

params.scale=1.0f/sqrtf(HEAD_DIM);

params.causal=true;

params.dtype=
transformer::DType::BF16;



std::cout<<"Launching FlashAttention...\n";


auto start=
std::chrono::high_resolution_clock::now();


transformer::launch_flash_attention(params);


CHECK(cudaDeviceSynchronize());


auto end=
std::chrono::high_resolution_clock::now();



double ms =
std::chrono::duration<double,std::milli>
(end-start).count();



std::vector<__nv_bfloat16> h_O(q_size);


CHECK(cudaMemcpy(
h_O.data(),
d_O,
sizeof(__nv_bfloat16)*q_size,
cudaMemcpyDeviceToHost));





// ==========================================================
// Reference
// ==========================================================


std::cout<<"Running CPU reference...\n";


reference_attention(
h_Q,
h_K,
h_V,
reference);





// ==========================================================
// Validate
// ==========================================================


float max_error=0;

float mean_error=0;

long errors=0;



for(size_t i=0;i<q_size;i++)
{

float gpu =
bf16_to_float(h_O[i]);


float ref =
reference[i];


float diff =
fabs(gpu-ref);



max_error =
fmax(max_error,diff);


mean_error+=diff;



if(diff>0.05f)
    errors++;

}



mean_error/=q_size;



std::cout<<"\n==============================\n";

std::cout<<"Validation\n";

std::cout<<"==============================\n";



std::cout
<<"Max error   : "
<<max_error<<"\n";


std::cout
<<"Mean error  : "
<<mean_error<<"\n";


std::cout
<<"Bad values  : "
<<errors<<"\n";



if(errors==0)
    std::cout<<"STATUS : PASS\n";
else
    std::cout<<"STATUS : FAIL\n";



// ==========================================================
// Performance
// ==========================================================


double flops =
2.0*
HEADS*
SEQ_LEN*
SEQ_LEN*
HEAD_DIM;


double tflops =
flops/(ms/1000.0)/1e12;


std::cout<<"\n==============================\n";
std::cout<<"Performance\n";
std::cout<<"==============================\n";


std::cout
<<"Latency : "
<<ms<<" ms\n";


std::cout
<<"TFLOPS  : "
<<tflops<<"\n";



cudaFree(d_Q);
cudaFree(d_K);
cudaFree(d_V);
cudaFree(d_O);
cudaFree(d_L);


return 0;

}