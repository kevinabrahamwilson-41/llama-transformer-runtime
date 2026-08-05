#ifndef LLAMA32_EMBEDDING_HPP
#define LLAMA32_EMBEDDING_HPP

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ======================================================
// Llama 3.2 1B Constants
// ======================================================

constexpr int VOCAB_SIZE   = 128256;
constexpr int HIDDEN_DIM   = 2048;
constexpr int SEQ_LEN      = 512;
constexpr int BLOCK_SIZE   = 256;

// ======================================================
// CUDA Error Checking
// ======================================================

#define CUDA_CHECK(call)                                              \
do {                                                                  \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        printf("CUDA Error %s:%d : %s\n",                             \
               __FILE__,                                               \
               __LINE__,                                               \
               cudaGetErrorString(err));                               \
        exit(EXIT_FAILURE);                                            \
    }                                                                 \
} while (0)

// ======================================================
// CUDA Launcher
//
// embedding_table : [VOCAB_SIZE][HIDDEN_DIM]
//
// tokens          : [SEQ_LEN]
//
// output          : [SEQ_LEN][HIDDEN_DIM]
//
// ======================================================

void launch_embedding(
    const int* d_tokens,
    const __nv_bfloat16* d_embedding_table,
    __nv_bfloat16* d_output,
    int seq_len,
    cudaStream_t stream = 0
);
#endif // LLAMA32_EMBEDDING_HPP