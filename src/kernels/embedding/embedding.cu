// ======================================================
// Embedding Lookup Kernel (Vectorized BF16x2)
//
// Grid
// -----
// grid.x  = SEQ_LEN
//
// Block
// ------
// block.x = 256 threads
//
// One block copies one embedding vector
//
// Embedding:
// [VOCAB_SIZE][2048]
//
// Output:
// [SEQ_LEN][2048]
//
// Each thread copies BF16x2 values.
// ======================================================
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include "embedding.hpp"
__global__
void embedding_kernel(
    const int* __restrict__ tokens,
    const __nv_bfloat16* __restrict__ embedding_table,
    __nv_bfloat16* __restrict__ output
){
    //----------------------------------------------------
    // One block = one token
    //----------------------------------------------------
    const int token_pos = blockIdx.x;
    if (token_pos >= SEQ_LEN)
        return;
    //----------------------------------------------------
    // Vocabulary index
    //----------------------------------------------------
    const int token_id = tokens[token_pos];
    //----------------------------------------------------
    // Locate embedding row
    //----------------------------------------------------
    const __nv_bfloat16* __restrict__ src =
        embedding_table +
        token_id * HIDDEN_DIM;
    //----------------------------------------------------
    // Output row
    //----------------------------------------------------
    __nv_bfloat16* __restrict__ dst =
        output +
        token_pos * HIDDEN_DIM;
    //----------------------------------------------------
    // Vectorized BF16 copy
    // 2048 BF16
    // = 1024 BF16x2 values
    //----------------------------------------------------
    const __nv_bfloat162* __restrict__ src2 =
        reinterpret_cast<const __nv_bfloat162*>(src);
    __nv_bfloat162* __restrict__ dst2 =
        reinterpret_cast<__nv_bfloat162*>(dst);
#pragma unroll
    for (int i = threadIdx.x;
         i < HIDDEN_DIM / 2;
         i += BLOCK_SIZE){
        dst2[i] = src2[i];
    }
}
// ======================================================
// Launch embedding kernel
// ======================================================

void launch_embedding(
    const int* d_tokens,
    const __nv_bfloat16* d_embedding_table,
    __nv_bfloat16* d_output,
    cudaStream_t stream
)
{
    constexpr dim3 block(BLOCK_SIZE);
    const dim3 grid(SEQ_LEN);
    embedding_kernel<<<
        grid,
        block,
        0,
        stream
    >>>(
        d_tokens,
        d_embedding_table,
        d_output
    );
}
int main()
{
    //--------------------------------------------------
    // Host memory
    //--------------------------------------------------
    std::vector<int> h_tokens(SEQ_LEN);

    const size_t embedding_elements =
        (size_t)VOCAB_SIZE * HIDDEN_DIM;

    std::vector<__nv_bfloat16> h_embedding_table(
        embedding_elements);

    std::vector<__nv_bfloat16> h_output(
        (size_t)SEQ_LEN * HIDDEN_DIM);

    //--------------------------------------------------
    // Initialize embedding table
    //--------------------------------------------------
    for (size_t i = 0; i < embedding_elements; i++)
    {
        float value = static_cast<float>(i % 1000) / 1000.0f;
        h_embedding_table[i] = __float2bfloat16(value);
    }

    //--------------------------------------------------
    // Random tokens
    //--------------------------------------------------
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(
        0,
        VOCAB_SIZE - 1);

    for (int i = 0; i < SEQ_LEN; i++)
        h_tokens[i] = dist(rng);

    //--------------------------------------------------
    // Device memory
    //--------------------------------------------------
    int* d_tokens;
    __nv_bfloat16* d_embedding_table;
    __nv_bfloat16* d_output;

    cudaMalloc(&d_tokens,
               sizeof(int) * SEQ_LEN);

    cudaMalloc(&d_embedding_table,
               sizeof(__nv_bfloat16) * embedding_elements);

    cudaMalloc(&d_output,
               sizeof(__nv_bfloat16) *
               SEQ_LEN * HIDDEN_DIM);

    //--------------------------------------------------
    // Copy to GPU
    //--------------------------------------------------
    cudaMemcpy(
        d_tokens,
        h_tokens.data(),
        sizeof(int) * SEQ_LEN,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_embedding_table,
        h_embedding_table.data(),
        sizeof(__nv_bfloat16) * embedding_elements,
        cudaMemcpyHostToDevice);

    //--------------------------------------------------
    // Launch
    //--------------------------------------------------
    launch_embedding(
        d_tokens,
        d_embedding_table,
        d_output,
        0);

    cudaDeviceSynchronize();

    //--------------------------------------------------
    // Copy result back
    //--------------------------------------------------
    cudaMemcpy(
        h_output.data(),
        d_output,
        sizeof(__nv_bfloat16) *
        SEQ_LEN * HIDDEN_DIM,
        cudaMemcpyDeviceToHost);

    //--------------------------------------------------
    // Verify
    //--------------------------------------------------
    bool passed = true;

    for (int token = 0; token < SEQ_LEN; token++)
    {
        int token_id = h_tokens[token];

        const __nv_bfloat16* expected =
            h_embedding_table.data() +
            token_id * HIDDEN_DIM;

        const __nv_bfloat16* actual =
            h_output.data() +
            token * HIDDEN_DIM;

        for (int j = 0; j < HIDDEN_DIM; j++)
        {
            float a = __bfloat162float(actual[j]);
            float b = __bfloat162float(expected[j]);

            if (fabs(a - b) > 1e-3f)
            {
                std::cout
                    << "Mismatch at token "
                    << token
                    << " element "
                    << j
                    << std::endl;

                passed = false;
                goto finish;
            }
        }
    }

finish:

    if (passed)
        std::cout << "PASS\n";
    else
        std::cout << "FAIL\n";

    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------
    cudaFree(d_tokens);
    cudaFree(d_embedding_table);
    cudaFree(d_output);

    return 0;
}