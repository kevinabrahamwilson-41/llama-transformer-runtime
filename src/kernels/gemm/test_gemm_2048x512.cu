#include "gemm_2048x512.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
void dump_fp32(
    const char* path,
    const std::vector<float>& data
)
{
    FILE* f = fopen(path,"w");

    for(size_t i=0;i<data.size();i++)
    {
        fprintf(f,"%.9g\n",data[i]);
    }

    fclose(f);
}

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess)                                   \
    {                                                         \
        printf("CUDA ERROR %s:%d -> %s\n",                   \
            __FILE__,                                        \
            __LINE__,                                        \
            cudaGetErrorString(err));                        \
        exit(EXIT_FAILURE);                                  \
    }                                                         \
} while(0)



void load_txt_bf16(
    const char* path,
    std::vector<__nv_bfloat16>& data,
    int elements
)
{
    std::ifstream file(path);

    if (!file)
    {
        printf("Cannot open %s\n", path);
        exit(EXIT_FAILURE);
    }


    data.resize(elements);


    for (int i = 0; i < elements; i++)
    {
        float value;

        file >> value;

        if (!file)
        {
            printf("Failed reading element %d from %s\n",
                   i,
                   path);
            exit(EXIT_FAILURE);
        }

        data[i] = __float2bfloat16(value);
    }


    printf("Loaded %d elements from %s\n",
           elements,
           path);
}



void dump_bf16(
    const char* path,
    const std::vector<__nv_bfloat16>& data
)
{
    FILE* f = fopen(path, "w");


    if (!f)
    {
        printf("Cannot open output %s\n", path);
        exit(EXIT_FAILURE);
    }


    for (size_t i = 0; i < data.size(); i++)
    {
        fprintf(
            f,
            "%.9g\n",
            __bfloat162float(data[i])
        );
    }


    fclose(f);


    printf("Dumped %zu elements to %s\n",
           data.size(),
           path);
}



int main()
{
    constexpr int TOKENS = 128;
    constexpr int HIDDEN = 2048;
    constexpr int K_DIM = 512;


    constexpr int A_ELEMENTS =
        TOKENS * HIDDEN;


    constexpr int B_ELEMENTS =
        HIDDEN * K_DIM;


    constexpr int C_ELEMENTS =
        TOKENS * K_DIM;



    printf("=============================\n");
    printf(" GEMM 2048x512 TEST\n");
    printf("=============================\n");



    // -----------------------------
    // Load A = attention_norm
    // -----------------------------

    std::vector<__nv_bfloat16> h_A;


    load_txt_bf16(
        "/tmp/random_A.txt",
        h_A,
        A_ELEMENTS
    );



    // -----------------------------
    // Load B = k_proj weights
    // -----------------------------

    std::vector<__nv_bfloat16> h_B;


    load_txt_bf16(
        "/tmp/random_B.txt",
        h_B,
        B_ELEMENTS
    );



    std::vector<float> h_C(
        C_ELEMENTS
    );



    __nv_bfloat16* d_A = nullptr;
    __nv_bfloat16* d_B = nullptr;
    float* d_C = nullptr;



    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            A_ELEMENTS * sizeof(__nv_bfloat16)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            B_ELEMENTS * sizeof(__nv_bfloat16)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            C_ELEMENTS * sizeof(float)
        )
    );



    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            A_ELEMENTS * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            B_ELEMENTS * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        )
    );



    printf("\nLaunching GEMM...\n");



    launch_gemm_2048x512(
        d_A,
        d_B,
        d_C,
        TOKENS,
        K_DIM,
        HIDDEN
    );



    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );



    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            C_ELEMENTS * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    // =============================
// CPU REFERENCE GEMM CHECK
// =============================

std::vector<float> cpu_C(C_ELEMENTS);

for(int i = 0; i < TOKENS; i++)
{
    for(int j = 0; j < K_DIM; j++)
    {
        float sum = 0.0f;

        for(int k = 0; k < HIDDEN; k++)
        {
            sum +=
                __bfloat162float(h_A[i * HIDDEN + k]) *
                __bfloat162float(h_B[k * K_DIM + j]);
        }

        cpu_C[i * K_DIM + j] = sum;
    }
}


float max_diff = 0.0f;
float mean_diff = 0.0f;
int worst_index = 0;


for(int i = 0; i < C_ELEMENTS; i++)
{
    float cuda_value =
        h_C[i];

    float diff =
        fabs(cpu_C[i] - cuda_value);


    mean_diff += diff;


    if(diff > max_diff)
    {
        max_diff = diff;
        worst_index = i;
    }
}


mean_diff /= C_ELEMENTS;
float max_rel = 0.0f;

for(int i = 0; i < C_ELEMENTS; i++)
{
    float cuda_value =
        h_C[i];

    float denom = fabs(cpu_C[i]);

    if(denom > 1e-6f)
    {
        float rel =
            fabs(cpu_C[i] - cuda_value) / denom;

        if(rel > max_rel)
            max_rel = rel;
    }
}

printf("max relative error: %f\n", max_rel);

printf("\n=============================\n");
printf(" CPU GEMM CHECK\n");
printf("=============================\n");

printf("max diff  : %f\n", max_diff);
printf("mean diff : %f\n", mean_diff);
printf("worst idx : %d\n", worst_index);

printf("CPU value : %f\n",
       cpu_C[worst_index]);

printf("CUDA value: %f\n",
       h_C[worst_index]);

   dump_fp32(
    "/tmp/test_cuda_k.txt",
    h_C
);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);



    printf("\nDONE\n");

    return 0;
}