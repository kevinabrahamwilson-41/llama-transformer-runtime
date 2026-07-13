#include "gemm_dispatch.hpp"

#include <stdexcept>
#include <sstream>

// Specialized GEMMs
#include "gemm_2048x2048.hpp"
#include "gemm_2048x512.hpp"
#include "gemm_2048x8192.hpp"
#include "gemm_8192x2048.hpp"
#include "gemm_2048x128256.hpp"

void gemm_dispatch(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M,
    int N,
    int K)
{
    // ===========================
    // Attention Q Projection
    // ===========================
    if (M == 2048 && N == 2048 && K == 2048)
    {
        launch_gemm_2048x2048(A, B, C, M, N, K);
        return;
    }

    // ===========================
    // Attention K/V Projection
    // ===========================
    if (M == 2048 && N == 512 && K == 2048)
    {
        launch_gemm_2048x512(A, B, C, M, N, K);
        return;
    }

    // ===========================
    // MLP Up Projection
    // ===========================
    if (M == 2048 && N == 8192 && K == 2048)
    {
        launch_gemm_2048x8192(A, B, C, M, N, K);
        return;
    }

    // ===========================
    // MLP Down Projection
    // ===========================
    if (M == 8192 && N == 2048 && K == 8192)
    {
        launch_gemm_8192x2048(A, B, C, M, N, K);
        return;
    }

    // ===========================
    // LM Head / Output Projection
    // ===========================
    if (M == 2048 && N == 2048 && K == 128256)
    {
        launch_gemm_2048x128256(A, B, C, M, N, K);
        return;
    }
    
    std::ostringstream oss;
    oss << "Unsupported GEMM shape: "
        << "M=" << M
        << ", N=" << N
        << ", K=" << K;

    throw std::runtime_error(oss.str());
}