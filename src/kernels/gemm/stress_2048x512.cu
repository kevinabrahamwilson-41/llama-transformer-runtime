// stress_2048x512.cu
//
// Benchmark driver for gemm_2048x512.cu
//
// Compile:
// nvcc -arch=sm_89 -O3 stress_2048x512.cu gemm_2048x512.cu -lcublas -o stress_2048x512
//
// Run:
// ./stress_2048x512

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>

// ---------------------------------------------------------
// Function implemented in gemm_2048x512.cu
// ---------------------------------------------------------
extern void launch_gemm_2048x512(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K
);

// ---------------------------------------------------------
// Configuration
// ---------------------------------------------------------
constexpr int HIDDEN_SIZE = 2048;

constexpr int WARMUP_ITERS = 20;
constexpr int BENCH_ITERS  = 100;

const int SEQ_LENGTHS[] = {
    32,
    64,
    128,
    256,
    512,
    1024,
    2048
};

// ---------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main()
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "GEMM 2048x512 BASELINE LATENCY BENCHMARK\n";
    std::cout << "============================================================\n";
    std::cout << "Matrix operation: [SeqLen x 2048] x [2048 x 512]\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Benchmark iterations: " << BENCH_ITERS << "\n";
    std::cout << "============================================================\n\n";

    // ---------------------------------------------------------
    // Create CUDA events for accurate GPU timing
    // ---------------------------------------------------------
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---------------------------------------------------------
    // Random number generator
    // ---------------------------------------------------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // ---------------------------------------------------------
    // Benchmark each sequence length
    // ---------------------------------------------------------
    std::cout
        << std::left
        << std::setw(12) << "Seq Len"
        << std::setw(16) << "M"
        << std::setw(16) << "N"
        << std::setw(16) << "K"
        << std::setw(20) << "Avg Latency (us)"
        << "\n";

    std::cout << "------------------------------------------------------------\n";

    for (int seq_len : SEQ_LENGTHS)
    {
        const int M = seq_len;
        const int N = 512;
        const int K = 2048;

        // -----------------------------------------------------
        // Matrix sizes
        //
        // A = [M x K]
        // B = [K x N]
        // C = [M x N]
        // -----------------------------------------------------
        size_t A_elements = static_cast<size_t>(M) * K;
        size_t B_elements = static_cast<size_t>(K) * N;
        size_t C_elements = static_cast<size_t>(M) * N;

        size_t A_bytes = A_elements * sizeof(__nv_bfloat16);
        size_t B_bytes = B_elements * sizeof(__nv_bfloat16);
        size_t C_bytes = C_elements * sizeof(__nv_bfloat16);

        // -----------------------------------------------------
        // Host allocations
        // -----------------------------------------------------
        std::vector<__nv_bfloat16> h_A(A_elements);
        std::vector<__nv_bfloat16> h_B(B_elements);

        // Fill input matrices
        for (size_t i = 0; i < A_elements; ++i)
        {
            h_A[i] = __float2bfloat16(dist(rng));
        }

        for (size_t i = 0; i < B_elements; ++i)
        {
            h_B[i] = __float2bfloat16(dist(rng));
        }

        // -----------------------------------------------------
        // Device allocations
        // -----------------------------------------------------
        __nv_bfloat16 *d_A = nullptr;
        __nv_bfloat16 *d_B = nullptr;
        __nv_bfloat16 *d_C = nullptr;

        CUDA_CHECK(cudaMalloc(&d_A, A_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, B_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, C_bytes));

        // Copy inputs to GPU
        CUDA_CHECK(cudaMemcpy(
            d_A,
            h_A.data(),
            A_bytes,
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_B,
            h_B.data(),
            B_bytes,
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemset(
            d_C,
            0,
            C_bytes
        ));

        // -----------------------------------------------------
        // Warmup
        // -----------------------------------------------------
        for (int i = 0; i < WARMUP_ITERS; ++i)
        {
            launch_gemm_2048x512(
                d_A,
                d_B,
                d_C,
                M,
                N,
                K
            );
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // -----------------------------------------------------
        // Timed benchmark
        // -----------------------------------------------------
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < BENCH_ITERS; ++i)
        {
            launch_gemm_2048x512(
                d_A,
                d_B,
                d_C,
                M,
                N,
                K
            );
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // -----------------------------------------------------
        // Calculate average latency
        // -----------------------------------------------------
        float total_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        ));

        float avg_us =
            (total_ms * 1000.0f) / BENCH_ITERS;

        // -----------------------------------------------------
        // Print result
        // -----------------------------------------------------
        std::cout
            << std::left
            << std::setw(12) << seq_len
            << std::setw(16) << M
            << std::setw(16) << N
            << std::setw(16) << K
            << std::setw(20)
            << std::fixed
            << std::setprecision(3)
            << avg_us
            << "\n";

        // -----------------------------------------------------
        // Cleanup
        // -----------------------------------------------------
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    // ---------------------------------------------------------
    // Cleanup events
    // ---------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Benchmark complete.\n";
    std::cout << "============================================================\n";

    return 0;
}
