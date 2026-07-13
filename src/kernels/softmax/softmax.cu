#include "softmax.hpp"
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

__forceinline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__forceinline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ __launch_bounds__(256,2) void softmax_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int rows
) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;
    __shared__ float s_row[SEQ_LEN];
    __shared__ float s_warp_results[WARPS];
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int base_offset = row_idx * SEQ_LEN;
    // Vectorized loads with bfloat162
    const __nv_bfloat162* input_vec = reinterpret_cast<const __nv_bfloat162*>(input);
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD / 2; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < SEQ_LEN / 2) {
            __nv_bfloat162 val = __ldg(&input_vec[base_offset / 2 + idx]);
            float2 val_f = __bfloat1622float2(val);
            s_row[idx * 2] = val_f.x;
            s_row[idx * 2 + 1] = val_f.y;
        }
    }
    __syncthreads();
    
    float thread_max = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < SEQ_LEN) {
            thread_max = fmaxf(thread_max, s_row[idx]);
        }
    }
    float warp_max = warp_reduce_max(thread_max);
    if (lane_id == 0) {
        s_warp_results[warp_id] = warp_max;
    }
    __syncthreads();
    
    float row_max = -FLT_MAX;
    if (tid < WARPS) {
        row_max = fmaxf(row_max, s_warp_results[tid]);
    }
    row_max = warp_reduce_max(row_max);
    // Broadcast row_max to the entire CTA
    if (tid == 0) {
        s_warp_results[0] = row_max;
    }
    __syncthreads();

    row_max = s_warp_results[0];
    
    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < SEQ_LEN) {
            float val = __expf(s_row[idx] - row_max);
            s_row[idx] = val;
            thread_sum += val;
        }
    }
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        s_warp_results[warp_id] = warp_sum;
    }
    __syncthreads();
    
    float row_sum = 0.0f;
    if (tid < WARPS) {
        row_sum = s_warp_results[tid];
    }
    row_sum = warp_reduce_sum(row_sum);
    // Broadcast row_sum to the entire CTA
    if (tid == 0) {
        s_warp_results[0] = row_sum;
    }
    __syncthreads();

    row_sum = s_warp_results[0];
    
    // Vectorized stores
    __nv_bfloat162* output_vec = reinterpret_cast<__nv_bfloat162*>(output);
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD / 2; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < SEQ_LEN / 2) {
            float2 val_f;
            val_f.x = s_row[idx * 2] / row_sum;
            val_f.y = s_row[idx * 2 + 1] / row_sum;
            output_vec[base_offset / 2 + idx] =
                __halves2bfloat162(
                    __float2bfloat16(val_f.x),
                    __float2bfloat16(val_f.y)
                );
        }
    }
}

cudaError_t launch_softmax_safe(const __nv_bfloat16* input, __nv_bfloat16* output, int rows) {
    if (input == nullptr) return cudaErrorInvalidValue;
    if (output == nullptr) return cudaErrorInvalidValue;
    if (rows <= 0) return cudaErrorInvalidValue;
    
    cudaError_t prev_err = cudaGetLastError();
    if (prev_err != cudaSuccess) {
        fprintf(stderr, "Previous CUDA error detected: %s\n", 
                cudaGetErrorString(prev_err));
        return prev_err;
    }
    
    softmax_kernel<<<rows, BLOCK_SIZE>>>(input, output, rows);
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", 
                cudaGetErrorString(launch_err));
        return launch_err;
    }
    
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", 
                cudaGetErrorString(sync_err));
        return sync_err;
    }
    
    return cudaSuccess;
}
int main() {
    constexpr int BATCH = 1;
    constexpr int ROWS = BATCH * HEADS * SEQ_LEN;
    constexpr size_t NUM_ELEMENTS = static_cast<size_t>(ROWS) * SEQ_LEN;
    std::cout << "Llama 3.2 1B Softmax Test\n";
    std::cout << "Rows           : " << ROWS << '\n';
    std::cout << "Sequence Length: " << SEQ_LEN << '\n';
    std::cout << "Total Elements : " << NUM_ELEMENTS << "\n\n";
    // ---------------- Host Memory ----------------
    std::vector<__nv_bfloat16> h_input(NUM_ELEMENTS);
    std::vector<__nv_bfloat16> h_output(NUM_ELEMENTS);
    // Random logits
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < NUM_ELEMENTS; i++) {
        h_input[i] = __float2bfloat16(dist(rng));
    }
    // ---------------- Device Memory ----------------
    __nv_bfloat16 *d_input = nullptr;
    __nv_bfloat16 *d_output = nullptr;
    if (cudaMalloc(&d_input, NUM_ELEMENTS * sizeof(__nv_bfloat16)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        return EXIT_FAILURE;
    }
    if (cudaMalloc(&d_output, NUM_ELEMENTS * sizeof(__nv_bfloat16)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        cudaFree(d_input);
        return EXIT_FAILURE;
    }
    cudaMemcpy(
        d_input,
        h_input.data(),
        NUM_ELEMENTS * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice
    );
    // ---------------- Launch Kernel ----------------
    cudaError_t err = launch_softmax_safe(d_input, d_output, ROWS);
    if (err != cudaSuccess) {
        std::cerr << "Kernel failed: "
                  << cudaGetErrorString(err)
                  << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
        return EXIT_FAILURE;
    }
    // ---------------- Copy Back ----------------
    cudaMemcpy(
        h_output.data(),
        d_output,
        NUM_ELEMENTS * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );
    // ---------------- Verify ----------------
    bool pass = true;
    for (int row = 0; row < ROWS; row++) {
        float sum = 0.0f;
        for (int j = 0; j < SEQ_LEN; j++) {
            sum += __bfloat162float(h_output[row * SEQ_LEN + j]);
        }
        if (std::fabs(sum - 1.0f) > 1e-2f) {
            std::cout << "Verification failed at row "
                      << row
                      << " (sum = "
                      << sum
                      << ")\n";
            pass = false;
            break;
        }
    }
    if (pass)
        std::cout << "Verification : PASS\n";
    else
        std::cout << "Verification : FAIL\n";
    // Print first 10 probabilities of first row
    std::cout << "\nFirst 10 outputs of row 0:\n";
    for (int i = 0; i < 10; i++) {
        std::cout
            << __bfloat162float(h_output[i])
            << " ";
    }
    std::cout << "\n";
    // ---------------- Cleanup ----------------
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}