// gemm_2048x2048.cu
// Compile: nvcc -arch=sm_89 -O3 gemm_2048x2048.cu -lcublas -o wmma_2048x2048
// Run: ./wmma_2048x2048
/*
Kernel	        Matrix	    Recommended CTA
gemm_2048x2048	2048×2048	32×64
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
using namespace nvcuda;
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
constexpr int CTA_M = 32;
constexpr int CTA_N = 64;

// =====================================================
// cp.async helper
// =====================================================

__device__ __forceinline__
void cp_async_wait_1()
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "cp.async.wait_group 1;\n"
        :
        :
        : "memory"
    );
#endif
}
__device__ __forceinline__
void cp_async_b16(
    __nv_bfloat16* smem,
    const __nv_bfloat16* gmem
)
{
#if __CUDA_ARCH__ >= 800

    uint32_t smem_addr =
        static_cast<uint32_t>(
            __cvta_generic_to_shared(smem)
        );

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        :
        "r"(smem_addr),
        "l"(gmem)
    );

#else

    *((uint4*)smem) = *((const uint4*)gmem);

#endif
}

__device__ __forceinline__
void cp_async_commit()
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "cp.async.commit_group;\n"
        :
        :
        : "memory"
    );
#endif
}

__device__ __forceinline__
void cp_async_wait()
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "cp.async.wait_group 0;\n"
        :
        :
        : "memory"
    );
#endif
}

// ============== BF16 WMMA GEMM WITH SHARED MEMORY + CP.ASYNC ==============
__global__
void bf16_tensorcore_gemm_2048x2048(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K
) {
    constexpr int AS_STRIDE = WMMA_K + 8;
    constexpr int BS_STRIDE = CTA_N + 8;
    __shared__ __nv_bfloat16 As[2][CTA_M][AS_STRIDE];
    __shared__ __nv_bfloat16 Bs[2][WMMA_K][BS_STRIDE];
    constexpr int C_STRIDE = WMMA_N;
    __shared__ float smem_C[8][WMMA_M][WMMA_N];
    int warp = threadIdx.x / 32;
    int cta_row = blockIdx.y;
    int cta_col = blockIdx.x;
    int cta_row_start = cta_row * CTA_M;
    int cta_col_start = cta_col * CTA_N;
    int warp_row = warp / 4;
    int warp_col = warp % 4;
    int tile_row = cta_row_start + warp_row * WMMA_M;
    int tile_col = cta_col_start + warp_col * WMMA_N;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    int stage = 0;
    // Preload first K tile into buffer 0
    constexpr int COPY_ELEMS = 8;
    for (int idx = threadIdx.x;
        idx < (CTA_M * WMMA_K) / COPY_ELEMS;
        idx += blockDim.x)
    {
        int offset = idx * COPY_ELEMS;
        int r = offset / WMMA_K;
        int c = offset % WMMA_K;
        if (cta_row_start + r < M)
        {
            cp_async_b16(
                &As[0][r][c],
                &A[(cta_row_start + r) * K + c]
            );
        }
        else
        {
            for (int i = 0; i < COPY_ELEMS; ++i)
            {
                As[0][r][c + i] =
                    __float2bfloat16(0.0f);
            }
        }
    }
    for (int idx = threadIdx.x;
        idx < (WMMA_K * CTA_N) / COPY_ELEMS;
        idx += blockDim.x)
    {
            int offset = idx * COPY_ELEMS;
            int r = offset / CTA_N;
            int c = offset % CTA_N;
            // Bounds-check B preload — last CTA may go out of range
            if (cta_col_start + c + COPY_ELEMS <= N) {
                cp_async_b16(
                    &Bs[0][r][c],
                    &B[r * N + (cta_col_start + c)]
                );
            } else {
                // zero pad the tail
                for (int i = 0; i < COPY_ELEMS; ++i) {
                    Bs[0][r][c + i] = __float2bfloat16(0.0f);
                }
            }
        }
        cp_async_commit();
        cp_async_wait();
        __syncthreads();
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                __nv_bfloat16, wmma::row_major> b_frag;
        for (int k = 0; k < K; k += WMMA_K)
    {
        const __nv_bfloat16* tile_A =
            &As[stage][warp_row * WMMA_M][0];

        const __nv_bfloat16* tile_B =
            &Bs[stage][0][warp_col * WMMA_N];
        // Load current tile from shared memory
        wmma::load_matrix_sync(
            a_frag,
            tile_A,
            AS_STRIDE
        );
        wmma::load_matrix_sync(
            b_frag,
            tile_B,
            BS_STRIDE
        );
        // ==============================
        // START ASYNC COPY OF NEXT TILE
        // ==============================
        if (k + WMMA_K < K)
        {
            int next_stage = stage ^ 1;
            // A copy
            for (int idx = threadIdx.x;
                idx < (CTA_M * WMMA_K) / COPY_ELEMS;
                idx += blockDim.x)
            {
                int offset = idx * COPY_ELEMS;
                int r = offset / WMMA_K;
                int c = offset % WMMA_K;
                if (cta_row_start + r < M)
                {
                    cp_async_b16(
                        &As[next_stage][r][c],
                        &A[
                            (cta_row_start + r) * K +
                            (k + WMMA_K + c)
                        ]
                    );
                }
                else
                {
                    for (int i = 0; i < COPY_ELEMS; ++i)
                    {
                        As[next_stage][r][c + i] =
                            __float2bfloat16(0.0f);
                    }
                }
            }
            // B copy
            for (int idx = threadIdx.x;
                idx < (WMMA_K * CTA_N) / COPY_ELEMS;
                idx += blockDim.x)
            {
                int offset = idx * COPY_ELEMS;
                int r = offset / CTA_N;
                int c = offset % CTA_N;
                if(
                    cta_col_start + c + COPY_ELEMS <= N &&
                    k + WMMA_K + r < K
                ){
                    cp_async_b16(
                        &Bs[next_stage][r][c],
                        &B[(k + WMMA_K + r)*N + cta_col_start+c]
                    );
                }
                else{
                    for(int i=0;i<COPY_ELEMS;i++){
                        Bs[next_stage][r][c+i] =
                            __float2bfloat16(0.0f);
                    }
                }
            }
            cp_async_commit();
        }
        // ==============================
        // COMPUTE CURRENT TILE
        // ==============================
        wmma::mma_sync(
            acc,
            a_frag,
            b_frag,
            acc
        );
        // ==============================
        // WAIT FOR NEXT TILE
        // ==============================

        if (k + WMMA_K < K)
        {
            cp_async_wait();
            __syncthreads();
            stage ^= 1;
        }
    }
    float* tile_C = &smem_C[warp][0][0];
    wmma::store_matrix_sync(
        tile_C,
        acc,
        WMMA_N,
        wmma::mem_row_major
    );
    __syncwarp();
    int lane = threadIdx.x % 32;
    for (int i = lane; i < WMMA_M * WMMA_N; i += 32)
    {
        int r = i / WMMA_N;
        int c = i % WMMA_N;
        if (
            tile_row + r < M &&
            tile_col + c < N
        )
        {
            C[
                (tile_row + r) * N +
                tile_col + c
            ] =
                __float2bfloat16(
                    tile_C[r * C_STRIDE + c]
                );
        }
    }
}
void launch_gemm_2048x2048(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K)
{
    dim3 block(256);

    dim3 grid(
        (N + CTA_N - 1) / CTA_N,
        (M + CTA_M - 1) / CTA_M
    );

    bf16_tensorcore_gemm_2048x2048<<<grid, block>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf(
            "GEMM ERROR: %s\n",
            cudaGetErrorString(err)
        );
    }
}
int main()
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;

    const size_t bytes_A =
        static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);

    const size_t bytes_B =
        static_cast<size_t>(K) * N * sizeof(__nv_bfloat16);

    const size_t bytes_C =
        static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);

    std::cout << "========================================\n";
    std::cout << "BF16 Tensor Core GEMM 2048x2048\n";
    std::cout << "A: " << M << "x" << K << "\n";
    std::cout << "B: " << K << "x" << N << "\n";
    std::cout << "C: " << M << "x" << N << "\n";
    std::cout << "========================================\n";

    // -------------------------------------------------
    // Allocate GPU memory
    // -------------------------------------------------

    __nv_bfloat16 *d_A = nullptr;
    __nv_bfloat16 *d_B = nullptr;
    __nv_bfloat16 *d_C = nullptr;

    cudaError_t err;

    err = cudaMalloc(&d_A, bytes_A);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_A failed: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }

    err = cudaMalloc(&d_B, bytes_B);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_B failed: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }

    err = cudaMalloc(&d_C, bytes_C);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_C failed: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // -------------------------------------------------
    // Initialize host matrices
    // -------------------------------------------------

    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(K * N);

    for (size_t i = 0; i < h_A.size(); ++i)
        h_A[i] = __float2bfloat16(0.01f);

    for (size_t i = 0; i < h_B.size(); ++i)
        h_B[i] = __float2bfloat16(0.01f);

    // -------------------------------------------------
    // Copy inputs to GPU
    // -------------------------------------------------

    cudaMemcpy(
        d_A,
        h_A.data(),
        bytes_A,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B.data(),
        bytes_B,
        cudaMemcpyHostToDevice
    );

    cudaMemset(d_C, 0, bytes_C);

    // -------------------------------------------------
    // Warm-up
    // -------------------------------------------------

    std::cout << "Warming up...\n";

    for (int i = 0; i < 10; ++i)
    {
        launch_gemm_2048x2048(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Warmup failed: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // -------------------------------------------------
    // Benchmark
    // -------------------------------------------------

    constexpr int ITERATIONS = 100;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Running " << ITERATIONS
              << " iterations...\n";

    cudaEventRecord(start);

    for (int i = 0; i < ITERATIONS; ++i)
    {
        launch_gemm_2048x2048(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float total_ms = 0.0f;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    );

    float avg_ms =
        total_ms / ITERATIONS;

    double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    double tflops =
        (flops / (avg_ms * 1e-3)) / 1e12;

    std::cout << "\n========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n";

    std::cout << "Total time : "
              << total_ms
              << " ms\n";

    std::cout << "Average    : "
              << avg_ms
              << " ms\n";

    std::cout << "TFLOPS     : "
              << tflops
              << "\n";

    // -------------------------------------------------
    // Final synchronization
    // -------------------------------------------------

    cudaDeviceSynchronize();

    // -------------------------------------------------
    // Cleanup
    // -------------------------------------------------

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Done.\n";

    return 0;
}