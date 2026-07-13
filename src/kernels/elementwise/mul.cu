// ======================================================
// Elementwise Multiply Kernel (Vectorized BF16x2)
//
// Computes:
//
// output = a * b
//
// Shapes
// ------
// a      : [512][8192]
// b      : [512][8192]
// output : [512][8192]
//
// Total Elements
// --------------
// 512 * 8192 = 4,194,304 BF16
//
// Vectorized as
//
// 2,097,152 BF16x2 values
//
// ======================================================

#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include "mul.hpp"
__global__
void elementwise_mul_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ output
)
{
    //----------------------------------------------------
    // BF16x2 pointers
    //----------------------------------------------------
    const __nv_bfloat162* __restrict__ a2 =
        reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* __restrict__ b2 =
        reinterpret_cast<const __nv_bfloat162*>(b);
    __nv_bfloat162* __restrict__ out2 =
        reinterpret_cast<__nv_bfloat162*>(output);
    //----------------------------------------------------
    // Total BF16x2 values
    //----------------------------------------------------
    constexpr int TOTAL =
        (SEQ_LEN * FFN_DIM) / 2;
    //----------------------------------------------------
    // Grid-stride loop
    //----------------------------------------------------
    for (
        int idx =
            blockIdx.x * blockDim.x +
            threadIdx.x;
        idx < TOTAL;
        idx +=
            blockDim.x * gridDim.x
    )
    {
        __nv_bfloat162 va = a2[idx];
        __nv_bfloat162 vb = b2[idx];
        out2[idx] =
            __hmul2(va, vb);
    }
}
void launch_elementwise_mul(
    const __nv_bfloat16* d_a,
    const __nv_bfloat16* d_b,
    __nv_bfloat16* d_output,
    cudaStream_t stream
)
{
    constexpr int TOTAL =
        (SEQ_LEN * FFN_DIM) / 2;

    const int grid =
        (TOTAL + BLOCK_SIZE - 1) / BLOCK_SIZE;

    elementwise_mul_kernel<<<
        grid,
        BLOCK_SIZE,
        0,
        stream
    >>>(
        d_a,
        d_b,
        d_output
    );

    CUDA_CHECK(cudaGetLastError());
}
int main()
{
    printf("===== Elementwise Multiply Test =====\n");

    constexpr int ELEMENTS =
        SEQ_LEN * FFN_DIM;

    const size_t bytes =
        ELEMENTS * sizeof(__nv_bfloat16);

    //--------------------------------------------------
    // Host
    //--------------------------------------------------

    __nv_bfloat16* h_a =
        new __nv_bfloat16[ELEMENTS];

    __nv_bfloat16* h_b =
        new __nv_bfloat16[ELEMENTS];

    __nv_bfloat16* h_out =
        new __nv_bfloat16[ELEMENTS];

    __nv_bfloat16* h_ref =
        new __nv_bfloat16[ELEMENTS];

    srand(1234);

    for(int i = 0; i < ELEMENTS; i++)
    {
        float x =
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        float y =
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        h_a[i] =
            __float2bfloat16(x);

        h_b[i] =
            __float2bfloat16(y);

        h_ref[i] =
            __float2bfloat16(x * y);
    }

    //--------------------------------------------------
    // Device
    //--------------------------------------------------

    __nv_bfloat16* d_a;
    __nv_bfloat16* d_b;
    __nv_bfloat16* d_out;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a,
        bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b,
        bytes,
        cudaMemcpyHostToDevice));

    //--------------------------------------------------
    // Launch
    //--------------------------------------------------

    launch_elementwise_mul(
        d_a,
        d_b,
        d_out,
        0);

    CUDA_CHECK(cudaDeviceSynchronize());

    //--------------------------------------------------
    // Copy back
    //--------------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        h_out,
        d_out,
        bytes,
        cudaMemcpyDeviceToHost));

    //--------------------------------------------------
    // Verify
    //--------------------------------------------------

    float max_error = 0.0f;
    int errors = 0;

    for(int i = 0; i < ELEMENTS; i++)
    {
        float gpu =
            __bfloat162float(h_out[i]);

        float cpu =
            __bfloat162float(h_ref[i]);

        float err =
            fabsf(gpu - cpu);

        if(err > max_error)
            max_error = err;

        if(err > 1e-2f)
            errors++;
    }

    printf("Max error : %.8f\n", max_error);
    printf("Errors    : %d / %d\n", errors, ELEMENTS);

    if(errors == 0)
        printf("PASS\n");
    else
        printf("FAIL\n");

    printf("\nFirst 8 outputs:\n");

    for(int i = 0; i < 8; i++)
    {
        printf("%f\n",
            __bfloat162float(h_out[i]));
    }

    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    delete[] h_a;
    delete[] h_b;
    delete[] h_out;
    delete[] h_ref;

    return 0;
}