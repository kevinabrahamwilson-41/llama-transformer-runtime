#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "rmsnorm.hpp"
int main()
{
    constexpr int ROWS = 512;
    constexpr int HIDDEN = 2048;
    constexpr float EPS = 1e-5f;

    std::cout << "=============================================\n";
    std::cout << "      Llama 3.2 CUDA RMSNorm Validation\n";
    std::cout << "=============================================\n\n";

    //--------------------------------------------------
    // Host Memory
    //--------------------------------------------------
    std::vector<__nv_bfloat16> h_input(
        ROWS * HIDDEN);

    std::vector<__nv_bfloat16> h_weight(
        HIDDEN);

    std::vector<__nv_bfloat16> h_output(
        ROWS * HIDDEN);

    //--------------------------------------------------
    // Initialize Input
    //--------------------------------------------------
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &x : h_input)
        x = __float2bfloat16(dist(rng));

    for (auto &x : h_weight)
        x = __float2bfloat16(1.0f);

    //--------------------------------------------------
    // Device Memory
    //--------------------------------------------------
    __nv_bfloat16 *d_input;
    __nv_bfloat16 *d_weight;
    __nv_bfloat16 *d_output;

    cudaMalloc(&d_input,
               sizeof(__nv_bfloat16) * ROWS * HIDDEN);

    cudaMalloc(&d_weight,
               sizeof(__nv_bfloat16) * HIDDEN);

    cudaMalloc(&d_output,
               sizeof(__nv_bfloat16) * ROWS * HIDDEN);

    cudaMemcpy(
        d_input,
        h_input.data(),
        sizeof(__nv_bfloat16) * ROWS * HIDDEN,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_weight,
        h_weight.data(),
        sizeof(__nv_bfloat16) * HIDDEN,
        cudaMemcpyHostToDevice);

    //--------------------------------------------------
    // Launch Kernel
    //--------------------------------------------------
    std::cout << "Launching RMSNorm kernel...\n";

    rmsnorm_launch<2048>(
        d_input,
        d_weight,
        d_output,
        ROWS,
        EPS,
        256);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_output.data(),
        d_output,
        sizeof(__nv_bfloat16) * ROWS * HIDDEN,
        cudaMemcpyDeviceToHost);

    //--------------------------------------------------
    // CPU Reference Validation
    //--------------------------------------------------
    bool passed = true;

    for (int row = 0; row < ROWS && passed; row++)
    {
        float sum_sq = 0.0f;

        for (int j = 0; j < HIDDEN; j++)
        {
            float x =
                __bfloat162float(
                    h_input[row * HIDDEN + j]);

            sum_sq += x * x;
        }

        float rms =
            rsqrtf(sum_sq / HIDDEN + EPS);

        for (int j = 0; j < HIDDEN; j++)
        {
            float expected =
                __bfloat162float(
                    h_input[row * HIDDEN + j])
                * rms;

            float actual =
                __bfloat162float(
                    h_output[row * HIDDEN + j]);

            if (fabs(expected - actual) > 5e-2f)
            {
                std::cout
                    << "\nMismatch at Row "
                    << row
                    << ", Element "
                    << j
                    << "\nExpected : "
                    << expected
                    << "\nActual   : "
                    << actual
                    << "\n";

                passed = false;
                break;
            }
        }
    }

    //--------------------------------------------------
    // Results
    //--------------------------------------------------
    std::cout << "\nRows        : " << ROWS << '\n';
    std::cout << "Hidden Size : " << HIDDEN << '\n';
    std::cout << "Block Size  : 256\n\n";

    std::cout << "First Output Values:\n";

    for (int i = 0; i < 8; i++)
        std::cout
            << __bfloat162float(h_output[i])
            << " ";

    std::cout << "\n\n";

    if (passed)
        std::cout
            << "RMSNorm Validation : PASSED\n";
    else
        std::cout
            << "RMSNorm Validation : FAILED\n";

    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    return 0;
}