#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "embedding.hpp"
int main()
{
    std::cout << "=============================================\n";
    std::cout << "     Llama 3.2 CUDA Embedding Validation\n";
    std::cout << "=============================================\n\n";

    //--------------------------------------------------
    // Host Memory
    //--------------------------------------------------
    std::vector<int> h_tokens(SEQ_LEN);

    const size_t embedding_elements =
        static_cast<size_t>(VOCAB_SIZE) * HIDDEN_DIM;

    std::vector<__nv_bfloat16> h_embedding_table(
        embedding_elements);

    std::vector<__nv_bfloat16> h_output(
        static_cast<size_t>(SEQ_LEN) * HIDDEN_DIM);

    //--------------------------------------------------
    // Initialize Embedding Table
    //--------------------------------------------------
    std::cout << "Initializing embedding table...\n";

    for (size_t i = 0; i < embedding_elements; i++)
    {
        float value =
            static_cast<float>(i % 1000) / 1000.0f;

        h_embedding_table[i] =
            __float2bfloat16(value);
    }

    //--------------------------------------------------
    // Generate Random Tokens
    //--------------------------------------------------
    std::cout << "Generating random token IDs...\n";

    std::mt19937 rng(1234);

    std::uniform_int_distribution<int> token_dist(
        0,
        VOCAB_SIZE - 1);

    for (int i = 0; i < SEQ_LEN; i++)
        h_tokens[i] = token_dist(rng);

    //--------------------------------------------------
    // Allocate Device Memory
    //--------------------------------------------------
    int* d_tokens = nullptr;
    __nv_bfloat16* d_embedding_table = nullptr;
    __nv_bfloat16* d_output = nullptr;

    cudaMalloc(
        &d_tokens,
        sizeof(int) * SEQ_LEN);

    cudaMalloc(
        &d_embedding_table,
        sizeof(__nv_bfloat16) *
        embedding_elements);

    cudaMalloc(
        &d_output,
        sizeof(__nv_bfloat16) *
        SEQ_LEN * HIDDEN_DIM);

    //--------------------------------------------------
    // Copy Data To GPU
    //--------------------------------------------------
    cudaMemcpy(
        d_tokens,
        h_tokens.data(),
        sizeof(int) * SEQ_LEN,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_embedding_table,
        h_embedding_table.data(),
        sizeof(__nv_bfloat16) *
        embedding_elements,
        cudaMemcpyHostToDevice);

    //--------------------------------------------------
    // Launch Kernel
    //--------------------------------------------------
    std::cout << "Launching embedding kernel...\n";
    launch_embedding(
        d_tokens,
        d_embedding_table,
        d_output,
        0);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cout << "Launch Error : "
                << cudaGetErrorString(err)
                << std::endl;
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cout << "Runtime Error : "
                << cudaGetErrorString(err)
                << std::endl;
    }

    //--------------------------------------------------
    // Copy Back
    //--------------------------------------------------
    cudaMemcpy(
        h_output.data(),
        d_output,
        sizeof(__nv_bfloat16) *
        SEQ_LEN * HIDDEN_DIM,
        cudaMemcpyDeviceToHost);

    //--------------------------------------------------
    // Validate
    //--------------------------------------------------
    bool passed = true;

    for (int token = 0; token < SEQ_LEN; token++)
    {
        int token_id = h_tokens[token];

        for (int j = 0; j < HIDDEN_DIM; j++)
        {
            float expected =
                __bfloat162float(
                    h_embedding_table[
                        token_id * HIDDEN_DIM + j]);

            float actual =
                __bfloat162float(
                    h_output[
                        token * HIDDEN_DIM + j]);

            if (fabs(expected - actual) > 1e-3f)
            {
                std::cout
                    << "\nValidation FAILED\n\n";

                std::cout
                    << "Token      : "
                    << token
                    << "\n";

                std::cout
                    << "Element    : "
                    << j
                    << "\n";

                std::cout
                    << "Expected   : "
                    << expected
                    << "\n";

                std::cout
                    << "Actual     : "
                    << actual
                    << "\n";

                passed = false;
                break;
            }
        }

        if (!passed)
            break;
    }

    //--------------------------------------------------
    // Results
    //--------------------------------------------------
    std::cout << "\n=============================================\n";
    std::cout << "Kernel Configuration\n";
    std::cout << "=============================================\n";

    std::cout
        << "Vocabulary Size : "
        << VOCAB_SIZE
        << "\n";

    std::cout
        << "Hidden Size     : "
        << HIDDEN_DIM
        << "\n";

    std::cout
        << "Sequence Length : "
        << SEQ_LEN
        << "\n";

    std::cout
        << "Block Size      : "
        << BLOCK_SIZE
        << "\n";

    std::cout
        << "\nSample Token IDs:\n";

    for (int i = 0; i < 8; i++)
        std::cout
            << h_tokens[i]
            << " ";

    std::cout
        << "\n\nFirst Embedding Values:\n";

    for (int i = 0; i < 8; i++)
    {
        std::cout
            << __bfloat162float(h_output[i])
            << " ";
    }

    std::cout << "\n\n";

    if (passed)
        std::cout
            << "Embedding Lookup Validation : PASSED\n";
    else
        std::cout
            << "Embedding Lookup Validation : FAILED\n";

    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------
    cudaFree(d_tokens);
    cudaFree(d_embedding_table);
    cudaFree(d_output);

    return 0;
}