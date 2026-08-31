// rmsnorm_pure.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cuda_bf16.h>
#include "rmsnorm.hpp"
// Warp-level reduction for sum
template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-wide reduction for sum
template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    extern __shared__ unsigned char _shared_mem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(_shared_mem);
    int lane = threadIdx.x & 31;      // lane within warp
    int wid  = threadIdx.x >> 5;      // warp ID
    // Number of warps in this block
    int numWarps = (blockDim.x + 31) / 32;
    // Step 1: Reduce within each warp
    val = warpReduceSum(val);
    // Step 2: Warp leaders write their sums to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    // Step 3: First warp loads the warp sums
    val = (lane < numWarps) ? shared[lane] : scalar_t(0);
    // Step 4: First warp reduces the warp sums
    if (wid == 0) {
        val = warpReduceSum(val);
        // Step 5: Store the final block sum
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    // Step 6: Every thread reads the final block sum
    return shared[0];
}

// RMSNorm forward kernel: one block per row
template<int HIDDEN>
__global__ void rmsnorm_fwd_kernel(
    const __nv_bfloat16* __restrict__ input,   // [rows, hidden]
    const __nv_bfloat16* __restrict__ weight,  // [hidden]
    __nv_bfloat16* __restrict__ output,        // [rows, hidden]
    int rows,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* row_in  = input  + row * HIDDEN;
    __nv_bfloat16*       row_out = output + row * HIDDEN;
    // Reinterpret the row as __nv_bfloat162 vectors.
    // HIDDEN must be divisible by 2.
    const uint4* row4 = reinterpret_cast<const uint4*>(row_in);
    // Compute sum of squares over hidden dimension
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = threadIdx.x; i < HIDDEN / 8; i += blockDim.x) {
        uint4 data = row4[i];
        const __nv_bfloat162* bf =
            reinterpret_cast<const __nv_bfloat162*>(&data);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 v =
                __bfloat1622float2(bf[j]);
            sum_sq += v.x * v.x;
            sum_sq += v.y * v.y;
        }
    }
    // Block reduction
    sum_sq = blockReduceSum(sum_sq);
    // Compute RMS reciprocal
    float rms_recip = rsqrtf(sum_sq / static_cast<float>(HIDDEN) + eps);
    // Vectorized pointers
    const uint4* weight4 =
        reinterpret_cast<const uint4*>(weight);
    uint4* out4 =
        reinterpret_cast<uint4*>(row_out);
    #pragma unroll
    for (int i = threadIdx.x; i < HIDDEN / 8; i += blockDim.x) {
        uint4 x_data = row4[i];
        uint4 w_data = weight4[i];
        const __nv_bfloat162* x =
            reinterpret_cast<const __nv_bfloat162*>(&x_data);
        const __nv_bfloat162* w =
            reinterpret_cast<const __nv_bfloat162*>(&w_data);
        __nv_bfloat162 result[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 xf = __bfloat1622float2(x[j]);
            float2 wf = __bfloat1622float2(w[j]);
            float2 yf;
            yf.x = xf.x * rms_recip * wf.x;
            yf.y = xf.y * rms_recip * wf.y;
            result[j] = __floats2bfloat162_rn(yf.x, yf.y);
        }
        out4[i] = *reinterpret_cast<uint4*>(result);
    }
    __syncthreads();
}

// Host helper: launch RMSNorm on [rows x hidden]
template<int HIDDEN>
void rmsnorm_launch(
        const __nv_bfloat16* d_in,
        const __nv_bfloat16* d_weight,
        __nv_bfloat16* d_out,
        int rows,
        float eps,
        int threads_per_block
    ) {
    int blocks = rows;
    int threads = threads_per_block;
    size_t shared_mem = ((threads_per_block + 31) / 32) * sizeof(float);
    rmsnorm_fwd_kernel<HIDDEN><<<blocks, threads, shared_mem, 0>>>(
        d_in,
        d_weight,
        d_out,
        rows,
        eps
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
template void rmsnorm_launch<2048>(
    const __nv_bfloat16*,
    const __nv_bfloat16*,
    __nv_bfloat16*,
    int,
    float,
    int
);
int main()
{
    // =====================================================
    // Configuration
    // =====================================================

    constexpr int HIDDEN = 2048;

    // Use a realistic number of rows/tokens.
    // For single-batch Llama inference, this can represent
    // the active token sequence.
    constexpr int ROWS = 128;

    constexpr int THREADS = 256;
    constexpr float EPS = 1e-5f;

    const size_t input_elements =
        static_cast<size_t>(ROWS) * HIDDEN;

    const size_t weight_elements =
        static_cast<size_t>(HIDDEN);

    const size_t bytes_input =
        input_elements * sizeof(__nv_bfloat16);

    const size_t bytes_weight =
        weight_elements * sizeof(__nv_bfloat16);

    const size_t bytes_output =
        input_elements * sizeof(__nv_bfloat16);

    std::printf("========================================\n");
    std::printf("RMSNorm Benchmark\n");
    std::printf("========================================\n");
    std::printf("Rows       : %d\n", ROWS);
    std::printf("Hidden     : %d\n", HIDDEN);
    std::printf("Threads    : %d\n", THREADS);
    std::printf("Epsilon    : %e\n", EPS);
    std::printf("========================================\n");

    // =====================================================
    // Host allocation
    // =====================================================

    std::vector<__nv_bfloat16> h_input(input_elements);
    std::vector<__nv_bfloat16> h_weight(weight_elements);

    // Simple deterministic initialization
    for (size_t i = 0; i < input_elements; ++i)
    {
        float value =
            0.01f * static_cast<float>((i % 100) - 50);

        h_input[i] = __float2bfloat16(value);
    }

    for (size_t i = 0; i < weight_elements; ++i)
    {
        h_weight[i] = __float2bfloat16(1.0f);
    }

    // =====================================================
    // Device allocation
    // =====================================================

    __nv_bfloat16* d_input = nullptr;
    __nv_bfloat16* d_weight = nullptr;
    __nv_bfloat16* d_output = nullptr;

    cudaError_t err;

    err = cudaMalloc(
        &d_input,
        bytes_input
    );

    if (err != cudaSuccess)
    {
        std::printf(
            "cudaMalloc d_input failed: %s\n",
            cudaGetErrorString(err)
        );
        return 1;
    }

    err = cudaMalloc(
        &d_weight,
        bytes_weight
    );

    if (err != cudaSuccess)
    {
        std::printf(
            "cudaMalloc d_weight failed: %s\n",
            cudaGetErrorString(err)
        );
        return 1;
    }

    err = cudaMalloc(
        &d_output,
        bytes_output
    );

    if (err != cudaSuccess)
    {
        std::printf(
            "cudaMalloc d_output failed: %s\n",
            cudaGetErrorString(err)
        );
        return 1;
    }

    // =====================================================
    // Copy input data to GPU
    // =====================================================

    cudaMemcpy(
        d_input,
        h_input.data(),
        bytes_input,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_weight,
        h_weight.data(),
        bytes_weight,
        cudaMemcpyHostToDevice
    );

    cudaMemset(
        d_output,
        0,
        bytes_output
    );

    // =====================================================
    // Warmup
    // =====================================================

    std::printf("\nWarming up...\n");

    constexpr int WARMUP_ITERATIONS = 10;

    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
    {
        rmsnorm_launch<HIDDEN>(
            d_input,
            d_weight,
            d_output,
            ROWS,
            EPS,
            THREADS
        );
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::printf(
            "Warmup failed: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_output);

        return 1;
    }

    // =====================================================
    // Benchmark
    // =====================================================

    constexpr int ITERATIONS = 100;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::printf(
        "Running %d iterations...\n",
        ITERATIONS
    );

    cudaEventRecord(start);

    for (int i = 0; i < ITERATIONS; ++i)
    {
        rmsnorm_launch<HIDDEN>(
            d_input,
            d_weight,
            d_output,
            ROWS,
            EPS,
            THREADS
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

    // =====================================================
    // Approximate memory traffic
    // =====================================================

    // Per invocation:
    //
    // Read input  : ROWS * HIDDEN * 2 bytes
    // Read weight : HIDDEN * 2 bytes
    // Write output: ROWS * HIDDEN * 2 bytes
    //
    // The actual DRAM traffic can differ because of cache
    // behavior, so this is only an approximate bandwidth
    // calculation.

    double bytes_per_run =
        static_cast<double>(bytes_input) +
        static_cast<double>(bytes_weight) +
        static_cast<double>(bytes_output);

    double bandwidth_gbs =
        bytes_per_run /
        (avg_ms * 1e-3) /
        1e9;

    // =====================================================
    // Results
    // =====================================================

    std::printf("\n");
    std::printf("========================================\n");
    std::printf("Results\n");
    std::printf("========================================\n");

    std::printf(
        "Total time       : %.4f ms\n",
        total_ms
    );

    std::printf(
        "Average time     : %.4f ms\n",
        avg_ms
    );

    std::printf(
        "Approx bandwidth : %.2f GB/s\n",
        bandwidth_gbs
    );

    std::printf(
        "Rows/sec         : %.2f M rows/s\n",
        (ROWS / (avg_ms * 1e-3)) / 1e6
    );

    // =====================================================
    // Cleanup
    // =====================================================

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    std::printf("Done.\n");

    return 0;
}
/*
run: nvcc -arch=sm_89 -lineinfo -O3 rmsnorm.cu -o rmsnorm
run: ./rmsnorm
run: sudo /usr/local/cuda/bin/ncu --page details --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section SchedulerStats ./rmsnorm
*/