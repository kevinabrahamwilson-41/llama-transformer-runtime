// wmma_test_llama3_2b.cu
// Compile: nvcc -arch=sm_89 -O3 wmma_test_llama3_2b.cu -lcublas -o wmma_test
// Run: ./wmma_test

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
constexpr int CTA_M = 64;
constexpr int CTA_N = 64;

// =====================================================
// cp.async helper
// =====================================================

__device__ __forceinline__
void cp_async_wait_1()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n");
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
    asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__
void cp_async_wait()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n");
#endif
}

// ============== YOUR KERNEL (baseline, no shared mem) ==============
__global__
void bf16_tensorcore_gemm(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M,
    int N,
    int K
) {
    constexpr int AS_STRIDE = WMMA_K + 8;
    constexpr int BS_STRIDE = CTA_N + 8;
    __shared__ __nv_bfloat16 As[2][CTA_M][AS_STRIDE];
    __shared__ __nv_bfloat16 Bs[2][WMMA_K][BS_STRIDE];
    int warp = threadIdx.x / 32;
    int cta_row = blockIdx.y;
    int cta_col = blockIdx.x;
    int cta_row_start = cta_row * CTA_M;
    int cta_col_start = cta_col * CTA_N;
    int warp_row = warp / 4;
    int warp_col = warp % 4;
    int tile_row = cta_row_start + warp_row * WMMA_M;
    int tile_col = cta_col_start + warp_col * WMMA_N;
    if (tile_row >= M || tile_col >= N)
        return;
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
        cp_async_b16(
            &As[0][r][c],
            &A[(cta_row_start+r)*K+c]
        );
    }
    for (int idx = threadIdx.x;
        idx < (WMMA_K * CTA_N) / COPY_ELEMS;
        idx += blockDim.x)
    {
            int offset = idx * COPY_ELEMS;
            int r = offset / CTA_N;
            int c = offset % CTA_N;
            cp_async_b16(
                &Bs[0][r][c],
                &B[r*N + cta_col_start+c]
            );
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
                cp_async_b16(
                    &As[next_stage][r][c],
                    &A[(cta_row_start+r)*K +
                    (k+WMMA_K+c)]
                );
            }
            // B copy
            for (int idx = threadIdx.x;
                idx < (WMMA_K * CTA_N) / COPY_ELEMS;
                idx += blockDim.x)
            {
                int offset = idx * COPY_ELEMS;
                int r = offset / CTA_N;
                int c = offset % CTA_N;
                cp_async_b16(
                    &Bs[next_stage][r][c],
                    &B[(k+WMMA_K+r)*N +
                    (cta_col_start+c)]
                );
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
    float* tile_C = C + tile_row * N + tile_col;
    wmma::store_matrix_sync(tile_C, acc, N, wmma::mem_row_major);
}

void launch_gemm(
    __nv_bfloat16* A,
    __nv_bfloat16* B,
    float* C,
    int M,
    int N,
    int K)
{
    dim3 block(256);        // 4 warps
    dim3 grid(
        (N + CTA_N - 1) / CTA_N,
        (M + CTA_M - 1) / CTA_M
    );
    bf16_tensorcore_gemm<<<grid, block>>>(A,B,C,M,N,K);
}

// ============== UTILS ==============
__nv_bfloat16 float2bf16(float x) {
    return __float2bfloat16(x);
}

float bf162float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

void init_matrix(__nv_bfloat16* mat, int rows, int cols, float scale = 0.1f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-scale, scale);
    for(int i = 0; i < rows * cols; i++) {
        mat[i] = float2bf16(dis(gen));
    }
}

void verify_results(float* gpu, float* ref, int size, float tol = 1e-3f) {
    int errors = 0;
    float max_err = 0.0f;
    for(int i = 0; i < size; i++) {
        float err = fabsf(gpu[i] - ref[i]);
        max_err = fmaxf(max_err, err);
        if(err > tol) errors++;
    }
    printf("Verification: max_err=%.6f, errors=%d/%d\n", max_err, errors, size);
}

// ============== MAIN ==============
int main() {
    const int M = 2048;  // Llama 3.2 1B hidden_dim
    const int N = 2048;
    const int K = 2048;
    printf("=== Llama 3.2 1B GEMM Benchmark (BF16, 2048x2048x2048) ===\n");
    printf("A: %d x %d, B: %d x %d, C: %d x %d\n", M, K, K, N, M, N);
    // Allocate host buffers (BF16 for A,B; float for C)
    size_t size_A = M * K * sizeof(__nv_bfloat16);
    size_t size_B = K * N * sizeof(__nv_bfloat16);
    size_t size_C = M * N * sizeof(float);
    __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(size_A);
    __nv_bfloat16* h_B = (__nv_bfloat16*)malloc(size_B);
    float* h_C_gpu = (float*)malloc(size_C);
    float* h_C_ref = (float*)malloc(size_C);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    // ============== cuBLAS reference ==============
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16BF, N,
        d_A, CUDA_R_16BF, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cudaMemcpy(h_C_ref, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // ============== YOUR KERNEL (baseline) ==============
    auto start = std::chrono::high_resolution_clock::now();
    launch_gemm(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * M * N * K) / (elapsed_ms * 1e6);
    double gb_s = ((M * K + K * N + M * N) * sizeof(__nv_bfloat16)) / (elapsed_ms * 1e6);
    printf("\n[Your WMMA Kernel] Time: %.3f ms, %.2f TFLOPS, %.2f GB/s\n",
           elapsed_ms, gflops / 1000.0, gb_s);
    verify_results(h_C_gpu, h_C_ref, M * N, 1e-2f);
    // ============== cuBLAS timing ==============
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);
    int warmup = 10, runs = 100;
    for(int i = 0; i < warmup; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
            d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(start_evt);
    for(int i = 0; i < runs; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
            d_B, CUDA_R_16BF, N, d_A, CUDA_R_16BF, K, &beta,
            d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float elapsed_cublas;
    cudaEventElapsedTime(&elapsed_cublas, start_evt, stop_evt);
    elapsed_cublas /= runs;
    double cublas_gflops = (2.0 * M * N * K) / (elapsed_cublas * 1e6);
    printf("\n[cuBLAS] Time: %.3f ms, %.2f TFLOPS\n", elapsed_cublas, cublas_gflops / 1000.0);
    printf("\nSpeedup vs cuBLAS: %.2fx\n", elapsed_cublas / elapsed_ms);
    // Cleanup
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    return 0;
}