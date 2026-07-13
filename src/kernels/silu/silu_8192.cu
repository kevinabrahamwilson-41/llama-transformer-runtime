#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
constexpr int BLOCK_SIZE = 256;
// ----------------------------
// SiLU activation (scalar float)
// ----------------------------
__device__ __forceinline__
float silu(float x) {
    return x * (1.0f / (1.0f + __expf(-x)));
}
// ----------------------------
// Vectorized BF16 SiLU
// Processes 2 BF16 values/thread
// ----------------------------
__global__
void silu_kernel_bf16_vec2(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    // vector path
    if (idx + 1 < elements) {
        __nv_bfloat162 in =
            *reinterpret_cast<const __nv_bfloat162*>(&input[idx]);

        float x0 = __bfloat162float(in.x);
        float x1 = __bfloat162float(in.y);

        float y0 = silu(x0);
        float y1 = silu(x1);

        __nv_bfloat162 out;
        out.x = __float2bfloat16(y0);
        out.y = __float2bfloat16(y1);

        *reinterpret_cast<__nv_bfloat162*>(&output[idx]) = out;
    }
    // tail element
    else if (idx < elements) {
        float x = __bfloat162float(input[idx]);
        float y = silu(x);
        output[idx] = __float2bfloat16(y);
    }
}

// ----------------------------
// Fused SiLU + multiplication:
// out = SiLU(gate) * up
// Vectorized BF16, 2 elements/thread
// ----------------------------
__device__ __forceinline__
float silu_mul(float gate, float up) {
    float s = gate * (1.0f / (1.0f + __expf(-gate)));
    return s * up;
}

__global__
void silu_mul_kernel_bf16_vec2(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    // vector path
    if (idx + 1 < elements) {
        const __nv_bfloat162* gate_vec =
            reinterpret_cast<const __nv_bfloat162*>(gate);
        const __nv_bfloat162* up_vec =
            reinterpret_cast<const __nv_bfloat162*>(up);
        __nv_bfloat162 g2 = gate_vec[idx / 2];
        __nv_bfloat162 u2 = up_vec[idx / 2];
        float g0 = __bfloat162float(g2.x);
        float g1 = __bfloat162float(g2.y);
        float u0 = __bfloat162float(u2.x);
        float u1 = __bfloat162float(u2.y);
        float h0 = silu_mul(g0, u0);
        float h1 = silu_mul(g1, u1);
        __nv_bfloat162 h2;
        h2.x = __float2bfloat16(h0);
        h2.y = __float2bfloat16(h1);

        *reinterpret_cast<__nv_bfloat162*>(&out[idx]) = h2;
    }
    // tail element
    else if (idx < elements) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        float h = silu_mul(g, u);
        out[idx] = __float2bfloat16(h);
    }
}

// ----------------------------
// CUDA error checking
// ----------------------------
#define CUDA_CHECK(call)                                    \
do {                                                        \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        printf("CUDA Error %s:%d : %s\n",                   \
               __FILE__, __LINE__,                          \
               cudaGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while (0)

// ----------------------------
// Launchers
// ----------------------------

// SiLU only: output = SiLU(input)
void launch_silu(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int seq_len,
    int intermediate = 8192
) {
    int elements = seq_len * intermediate;
    int threads = BLOCK_SIZE;
    int blocks = (elements + (threads * 2) - 1) / (threads * 2);

    silu_kernel_bf16_vec2<<<blocks, threads>>>(
        input, output, elements
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Fused SiLU + multiplication:
// out = SiLU(gate) * up
void launch_silu_mul(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    int seq_len,
    int intermediate = 8192
) {
    int elements = seq_len * intermediate;
    int threads = BLOCK_SIZE;
    int blocks = (elements + (threads * 2) - 1) / (threads * 2);

    silu_mul_kernel_bf16_vec2<<<blocks, threads>>>(
        gate, up, out, elements
    );
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------
// Example main() showing usage
// ----------------------------
int main() {
    // Example dimensions: Llama-style MLP
    // seq_len tokens, hidden -> intermediate = 8192
    int seq_len = 1;
    int intermediate = 8192;
    int elements = seq_len * intermediate;
    size_t bytes = elements * sizeof(__nv_bfloat16);

    // Host allocations (just for demo; normally you'd have real data)
    __nv_bfloat16* h_gate = new __nv_bfloat16[elements];
    __nv_bfloat16* h_up   = new __nv_bfloat16[elements];
    __nv_bfloat16* h_out  = new __nv_bfloat16[elements];
    printf("Host-side reference SiLU * up (first 8):\n");
    for (int i = 0; i < 8; ++i) {
        float x = static_cast<float>(i % 10) - 5.0f;
        x = x * 0.5f;
        float gate = x;
        float up = 1.0f;
        float s = gate * (1.0f / (1.0f + expf(-gate)));
        float y = s * up;
        printf("i=%d x=%.4f SiLU(x)*up=%.6f\n", i, x, y);
    }
    // Initialize with some dummy data
    for (int i = 0; i < elements; ++i) {
        float x = static_cast<float>(i % 10) - 5.0f; // [-5, 4] strictly non-zero otherwise all zeroes in the output 
        float gate = x * 0.5f;
        float up = 1.0f;
        h_gate[i] = __float2bfloat16(gate);
        h_up[i]   = __float2bfloat16(up);
    }

    // Device allocations
    __nv_bfloat16 *d_gate, *d_up, *d_out;
    CUDA_CHECK(cudaMalloc(&d_gate, bytes));
    CUDA_CHECK(cudaMalloc(&d_up, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_gate, h_gate, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up, bytes, cudaMemcpyHostToDevice));

    // Launch fused SiLU + mul: out = SiLU(gate) * up
    launch_silu_mul(d_gate, d_up, d_out, seq_len, intermediate);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Quick sanity check: print first few values
    printf("First 8 outputs (BF16 as float):\n");
    for (int i = 0; i < 8; ++i) {
        float v = __bfloat162float(h_out[i]);
        printf("out[%d] = %f\n", i, v);
    }

    // Cleanup
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);
    delete[] h_gate;
    delete[] h_up;
    delete[] h_out;

    return 0;
}