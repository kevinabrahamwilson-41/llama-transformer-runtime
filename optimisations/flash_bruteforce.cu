// ============================================================================
// bruteflash.cu
// RTX 4060 Ada Lovelace FlashAttention brute-force tuner
// BF16 ONLY
// ============================================================================
#include "../src/kernels/flashattention/flashattention.cu"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using transformer::flash_attention_fat_kernel;

#define BRUTE_CUDA_CHECK(call)                                             \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr,                                            \
                         "[CUDA ERROR] %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(err__));    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

static constexpr int WARP_SIZE = 32;

// ============================================================================
// Function 1: CUDA error-state reset
// ============================================================================

static void clear_cuda_errors()
{
    cudaGetLastError();
}

// ============================================================================
// Function 2: configuration descriptor
// ============================================================================

struct Config
{
    int block_m;
    int block_n;
    int d_head;
    int num_warps;
    bool causal;

    std::size_t shared_bytes;

    float latency_ms;
    float tflops;

    bool compiled;
    bool launched;
    bool numerically_valid;

    float max_error;
    float max_output_abs;

    int registers_per_thread;

    std::string failure_reason;
};

// ============================================================================
// Function 3: configuration constructor / validation
// ============================================================================

static Config make_config(
    int block_n,
    int d_head,
    int num_warps,
    bool causal)
{
    Config c{};

    c.num_warps = num_warps;
    c.block_m = 16 * num_warps;
    c.block_n = block_n;
    c.d_head = d_head;
    c.causal = causal;

    c.shared_bytes =
        2ULL *
        static_cast<std::size_t>(block_n) *
        static_cast<std::size_t>(d_head + 8) *
        sizeof(__nv_bfloat16);

    c.latency_ms = std::numeric_limits<float>::infinity();
    c.tflops = 0.0f;

    c.compiled = true;
    c.launched = false;
    c.numerically_valid = false;

    c.max_error = std::numeric_limits<float>::infinity();
    c.max_output_abs = 0.0f;

    c.registers_per_thread = -1;

    if (c.block_m > c.block_n) {
        c.compiled = false;
        c.failure_reason = "BLOCK_M > BLOCK_N";
    }

    if ((c.block_n % 16) != 0) {
        c.compiled = false;
        c.failure_reason = "BLOCK_N not divisible by 16";
    }

    if ((c.d_head % 16) != 0) {
        c.compiled = false;
        c.failure_reason = "D_HEAD not divisible by 16";
    }

    if (c.num_warps != 1 &&
        c.num_warps != 2 &&
        c.num_warps != 4 &&
        c.num_warps != 8) {
        c.compiled = false;
        c.failure_reason = "unsupported NUM_WARPS";
    }

    if (c.d_head != 64 && c.d_head != 128) {
        c.compiled = false;
        c.failure_reason = "unsupported D_HEAD";
    }

    return c;
}
// ============================================================================
// Function 4: launch one compile-time configuration
// ============================================================================

template <
    int BLOCK_M,
    int BLOCK_N,
    int D_HEAD,
    int NUM_WARPS,
    bool CAUSAL>
static void launch_config(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int q_seq_len,
    int kv_seq_len,
    int kv_stride,
    float scale,
    cudaStream_t stream)
{
    static_assert(BLOCK_M == 16 * NUM_WARPS);
    static_assert(BLOCK_N % 16 == 0);
    static_assert(D_HEAD % 16 == 0);
    static_assert(BLOCK_M <= BLOCK_N);

    constexpr int KV_STRIDE = D_HEAD + 8;

    dim3 block(WARP_SIZE, NUM_WARPS);

    dim3 grid(
        (q_seq_len + BLOCK_M - 1) / BLOCK_M,
        batch_size * num_heads);

    constexpr std::size_t SMEM =
        2ULL *
        BLOCK_N *
        KV_STRIDE *
        sizeof(__nv_bfloat16);

    flash_attention_fat_kernel<
        BLOCK_M,
        BLOCK_N,
        D_HEAD,
        NUM_WARPS,
        CAUSAL,
        __nv_bfloat16>
        <<<grid, block, SMEM, stream>>>(
            Q,
            K,
            V,
            O,
            LSE,
            q_seq_len,
            kv_seq_len,
            kv_stride,
            num_heads,
            num_kv_heads,
            scale);
}

// ============================================================================
// Function 5: dispatch every compile-time configuration
// ============================================================================

static bool dispatch_config(
    const Config& cfg,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int q_seq_len,
    int kv_seq_len,
    int kv_stride,
    float scale,
    cudaStream_t stream)
{
#define DISPATCH(BM, BN, DH, NW, CAUSAL_VALUE)                 \
    launch_config<BM, BN, DH, NW, CAUSAL_VALUE>(              \
        Q, K, V, O, LSE,                                     \
        batch_size, num_heads, num_kv_heads,                 \
        q_seq_len, kv_seq_len, kv_stride, scale, stream)

    if (cfg.d_head == 64) {

        if (cfg.num_warps == 1 && cfg.block_n == 16) {
            if (cfg.causal) DISPATCH(16,16,64,1,true);
            else            DISPATCH(16,16,64,1,false);
        }
        else if (cfg.num_warps == 1 && cfg.block_n == 32) {
            if (cfg.causal) DISPATCH(16,32,64,1,true);
            else            DISPATCH(16,32,64,1,false);
        }
        else if (cfg.num_warps == 1 && cfg.block_n == 48) {
            if (cfg.causal) DISPATCH(16,48,64,1,true);
            else            DISPATCH(16,48,64,1,false);
        }
        else if (cfg.num_warps == 1 && cfg.block_n == 64) {
            if (cfg.causal) DISPATCH(16,64,64,1,true);
            else            DISPATCH(16,64,64,1,false);
        }
        else if (cfg.num_warps == 1 && cfg.block_n == 80) {
            if (cfg.causal) DISPATCH(16,80,64,1,true);
            else            DISPATCH(16,80,64,1,false);
        }
        else if (cfg.num_warps == 1 && cfg.block_n == 96) {
            if (cfg.causal) DISPATCH(16,96,64,1,true);
            else            DISPATCH(16,96,64,1,false);
        }
        else if (cfg.num_warps == 1 && cfg.block_n == 112) {
            if (cfg.causal) DISPATCH(16,112,64,1,true);
            else            DISPATCH(16,112,64,1,false);
        }
        else if (cfg.num_warps == 1 && cfg.block_n == 128) {
            if (cfg.causal) DISPATCH(16,128,64,1,true);
            else            DISPATCH(16,128,64,1,false);
        }

        else if (cfg.num_warps == 2 && cfg.block_n == 32) {
            if (cfg.causal) DISPATCH(32,32,64,2,true);
            else            DISPATCH(32,32,64,2,false);
        }
        else if (cfg.num_warps == 2 && cfg.block_n == 48) {
            if (cfg.causal) DISPATCH(32,48,64,2,true);
            else            DISPATCH(32,48,64,2,false);
        }
        else if (cfg.num_warps == 2 && cfg.block_n == 64) {
            if (cfg.causal) DISPATCH(32,64,64,2,true);
            else            DISPATCH(32,64,64,2,false);
        }
        else if (cfg.num_warps == 2 && cfg.block_n == 80) {
            if (cfg.causal) DISPATCH(32,80,64,2,true);
            else            DISPATCH(32,80,64,2,false);
        }
        else if (cfg.num_warps == 2 && cfg.block_n == 96) {
            if (cfg.causal) DISPATCH(32,96,64,2,true);
            else            DISPATCH(32,96,64,2,false);
        }
        else if (cfg.num_warps == 2 && cfg.block_n == 112) {
            if (cfg.causal) DISPATCH(32,112,64,2,true);
            else            DISPATCH(32,112,64,2,false);
        }
        else if (cfg.num_warps == 2 && cfg.block_n == 128) {
            if (cfg.causal) DISPATCH(32,128,64,2,true);
            else            DISPATCH(32,128,64,2,false);
        }

        else if (cfg.num_warps == 4 && cfg.block_n >= 64) {
            switch (cfg.block_n) {
            case 64:
                if (cfg.causal) DISPATCH(64,64,64,4,true);
                else            DISPATCH(64,64,64,4,false);
                break;
            case 80:
                if (cfg.causal) DISPATCH(64,80,64,4,true);
                else            DISPATCH(64,80,64,4,false);
                break;
            case 96:
                if (cfg.causal) DISPATCH(64,96,64,4,true);
                else            DISPATCH(64,96,64,4,false);
                break;
            case 112:
                if (cfg.causal) DISPATCH(64,112,64,4,true);
                else            DISPATCH(64,112,64,4,false);
                break;
            case 128:
                if (cfg.causal) DISPATCH(64,128,64,4,true);
                else            DISPATCH(64,128,64,4,false);
                break;
            default:
                return false;
            }
        }

        else if (cfg.num_warps == 8 && cfg.block_n >= 128) {
            if (cfg.causal) DISPATCH(128,128,64,8,true);
            else            DISPATCH(128,128,64,8,false);
        }
        else {
            return false;
        }
    }

    else if (cfg.d_head == 128) {

        if (cfg.num_warps == 1 && cfg.block_n >= 16) {
            switch (cfg.block_n) {
            case 16:
                if (cfg.causal) DISPATCH(16,16,128,1,true);
                else            DISPATCH(16,16,128,1,false);
                break;
            case 32:
                if (cfg.causal) DISPATCH(16,32,128,1,true);
                else            DISPATCH(16,32,128,1,false);
                break;
            case 48:
                if (cfg.causal) DISPATCH(16,48,128,1,true);
                else            DISPATCH(16,48,128,1,false);
                break;
            case 64:
                if (cfg.causal) DISPATCH(16,64,128,1,true);
                else            DISPATCH(16,64,128,1,false);
                break;
            case 80:
                if (cfg.causal) DISPATCH(16,80,128,1,true);
                else            DISPATCH(16,80,128,1,false);
                break;
            case 96:
                if (cfg.causal) DISPATCH(16,96,128,1,true);
                else            DISPATCH(16,96,128,1,false);
                break;
            case 112:
                if (cfg.causal) DISPATCH(16,112,128,1,true);
                else            DISPATCH(16,112,128,1,false);
                break;
            case 128:
                if (cfg.causal) DISPATCH(16,128,128,1,true);
                else            DISPATCH(16,128,128,1,false);
                break;
            default:
                return false;
            }
        }

        else if (cfg.num_warps == 2 && cfg.block_n >= 32) {
            switch (cfg.block_n) {
            case 32:
                if (cfg.causal) DISPATCH(32,32,128,2,true);
                else            DISPATCH(32,32,128,2,false);
                break;
            case 48:
                if (cfg.causal) DISPATCH(32,48,128,2,true);
                else            DISPATCH(32,48,128,2,false);
                break;
            case 64:
                if (cfg.causal) DISPATCH(32,64,128,2,true);
                else            DISPATCH(32,64,128,2,false);
                break;
            case 80:
                if (cfg.causal) DISPATCH(32,80,128,2,true);
                else            DISPATCH(32,80,128,2,false);
                break;
            case 96:
                if (cfg.causal) DISPATCH(32,96,128,2,true);
                else            DISPATCH(32,96,128,2,false);
                break;
            case 112:
                if (cfg.causal) DISPATCH(32,112,128,2,true);
                else            DISPATCH(32,112,128,2,false);
                break;
            case 128:
                if (cfg.causal) DISPATCH(32,128,128,2,true);
                else            DISPATCH(32,128,128,2,false);
                break;
            default:
                return false;
            }
        }

        else if (cfg.num_warps == 4 && cfg.block_n >= 64) {
            switch (cfg.block_n) {
            case 64:
                if (cfg.causal) DISPATCH(64,64,128,4,true);
                else            DISPATCH(64,64,128,4,false);
                break;
            case 80:
                if (cfg.causal) DISPATCH(64,80,128,4,true);
                else            DISPATCH(64,80,128,4,false);
                break;
            case 96:
                if (cfg.causal) DISPATCH(64,96,128,4,true);
                else            DISPATCH(64,96,128,4,false);
                break;
            case 112:
                if (cfg.causal) DISPATCH(64,112,128,4,true);
                else            DISPATCH(64,112,128,4,false);
                break;
            case 128:
                if (cfg.causal) DISPATCH(64,128,128,4,true);
                else            DISPATCH(64,128,128,4,false);
                break;
            default:
                return false;
            }
        }

        else if (cfg.num_warps == 8 && cfg.block_n == 128) {
            if (cfg.causal) DISPATCH(128,128,128,8,true);
            else            DISPATCH(128,128,128,8,false);
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }

#undef DISPATCH

    return true;
}

// ============================================================================
// Function 6: benchmark one configuration
// ============================================================================

bool benchmark_config(
    Config& cfg,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int q_seq_len,
    int kv_seq_len,
    int kv_stride,
    float scale,
    cudaStream_t stream,
    int warmup,
    int iterations)
{
    if (!cfg.compiled)
        return false;

    clear_cuda_errors();

    // Warmup.
    for (int i = 0; i < warmup; ++i) {

        if (!dispatch_config(
                cfg, Q, K, V, O, LSE,
                batch_size,
                num_heads,
                num_kv_heads,
                q_seq_len,
                kv_seq_len,
                kv_stride,
                scale,
                stream)) {

            cfg.launched = false;
            cfg.failure_reason = "dispatch failed";
            return false;
        }
    }

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        cfg.launched = false;
        cfg.failure_reason = cudaGetErrorString(err);
        return false;
    }

    BRUTE_CUDA_CHECK(cudaStreamSynchronize(stream));

    // ------------------------------------------------------------------------
    // CUDA-event timing
    // ------------------------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    BRUTE_CUDA_CHECK(cudaEventCreate(&start));
    BRUTE_CUDA_CHECK(cudaEventCreate(&stop));

    BRUTE_CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < iterations; ++i) {

        if (!dispatch_config(
                cfg, Q, K, V, O, LSE,
                batch_size,
                num_heads,
                num_kv_heads,
                q_seq_len,
                kv_seq_len,
                kv_stride,
                scale,
                stream)) {

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cfg.launched = false;
            cfg.failure_reason = "dispatch failed during benchmark";
            return false;
        }
    }

    BRUTE_CUDA_CHECK(cudaEventRecord(stop, stream));
    BRUTE_CUDA_CHECK(cudaEventSynchronize(stop));

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        cfg.launched = false;
        cfg.failure_reason = cudaGetErrorString(err);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return false;
    }

    float total_ms = 0.0f;

    BRUTE_CUDA_CHECK(
        cudaEventElapsedTime(&total_ms, start, stop));

    cfg.latency_ms =
        total_ms / static_cast<float>(iterations);

    // Attention FLOPs:
    //
    // QK^T = 2 * Q * K * D
    // PV    = 2 * Q * K * D
    //
    // Total ~= 4 * Q * K * D
    //
    // We use the full theoretical workload here so that configurations
    // are compared using the same metric.

    const double flops =
        4.0 *
        static_cast<double>(batch_size) *
        static_cast<double>(num_heads) *
        static_cast<double>(q_seq_len) *
        static_cast<double>(kv_seq_len) *
        static_cast<double>(cfg.d_head);

    const double seconds =
        static_cast<double>(cfg.latency_ms) * 1.0e-3;

    cfg.tflops =
        static_cast<float>((flops / seconds) / 1.0e12);

    cfg.launched = true;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}
// ============================================================================
// MAIN
// ============================================================================
int main()
{
    std::cout
        << "============================================================\n"
        << " FlashAttention Bruteforce Tuner\n"
        << " NVIDIA RTX 4060 / Ada Lovelace\n"
        << " BF16 - Llama 3.2 1B GQA\n"
        << "============================================================\n\n";

    // ================================================================
    // FIXED LLAMA 3.2 1B CONFIGURATION
    // ================================================================

    constexpr int B = 1;
    constexpr int H = 32;
    constexpr int NUM_KV_HEADS = 8;
    constexpr int D_HEAD = 64;

    constexpr bool CAUSAL = true;

    constexpr int WARMUP = 20;
    constexpr int ITERATIONS = 100;

    const float scale =
        1.0f / std::sqrt(static_cast<float>(D_HEAD));

    // ================================================================
    // REAL WORKLOADS
    // ================================================================

    const int prefill_lengths[] = {
        128, 256, 512, 1024, 2048
    };

    const int decode_kv_lengths[] = {
        128, 256, 512, 1024, 2048
    };

    // ================================================================
    // KERNEL PARAMETERS TO SEARCH
    // ================================================================

    const int block_ns[] = {
        16, 32, 48, 64, 80, 96, 112, 128
    };

    const int num_warps_list[] = {
        1, 2, 4, 8
    };

    // ================================================================
    // DEVICE INFORMATION
    // ================================================================

    cudaDeviceProp prop{};
    BRUTE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout
        << "GPU               : " << prop.name << "\n"
        << "Compute capability : "
        << prop.major << "." << prop.minor << "\n"
        << "SM count           : "
        << prop.multiProcessorCount << "\n"
        << "Shared memory/SM   : "
        << (prop.sharedMemPerMultiprocessor / 1024)
        << " KB\n"
        << "Shared memory/CTA  : "
        << (prop.sharedMemPerBlock / 1024)
        << " KB\n"
        << "Max threads/CTA    : "
        << prop.maxThreadsPerBlock << "\n\n";

    if (prop.major != 8 || prop.minor != 9)
    {
        std::cerr
            << "[WARNING] Expected Ada Lovelace SM 8.9.\n";
    }

    cudaStream_t stream;
    BRUTE_CUDA_CHECK(cudaStreamCreate(&stream));

    // ================================================================
    // HELPER LAMBDA
    // ================================================================
    //
    // This runs the complete brute-force search for ONE workload.
    //
    // Example:
    //     run_workload(128, 128, "PREFILL");
    //     run_workload(1,   128, "DECODE");
    //
    // ================================================================

    auto run_workload =
        [&](int Q_LEN,
            int KV_LEN,
            const char* workload_name)
    {
        std::cout
            << "\n============================================================\n"
            << " " << workload_name
            << "   Q_LEN=" << Q_LEN
            << "   KV_LEN=" << KV_LEN
            << "\n============================================================\n";

        // ------------------------------------------------------------
        // Allocate tensors for THIS workload
        // ------------------------------------------------------------
        const std::size_t q_elements =
            static_cast<std::size_t>(B) *
            H *
            Q_LEN *
            D_HEAD;
        const std::size_t kv_elements =
            static_cast<std::size_t>(B) *
            NUM_KV_HEADS *
            KV_LEN *
            D_HEAD;
        const std::size_t out_elements = q_elements;
        const int KV_STRIDE = KV_LEN;
        __nv_bfloat16* d_Q = nullptr;
        __nv_bfloat16* d_K = nullptr;
        __nv_bfloat16* d_V = nullptr;
        __nv_bfloat16* d_O = nullptr;
        float* d_LSE = nullptr;

        BRUTE_CUDA_CHECK(
            cudaMalloc(
                &d_Q,
                q_elements * sizeof(__nv_bfloat16)));

        BRUTE_CUDA_CHECK(
            cudaMalloc(
                &d_K,
                kv_elements * sizeof(__nv_bfloat16)));

        BRUTE_CUDA_CHECK(
            cudaMalloc(
                &d_V,
                kv_elements * sizeof(__nv_bfloat16)));

        BRUTE_CUDA_CHECK(
            cudaMalloc(
                &d_O,
                out_elements * sizeof(__nv_bfloat16)));

        BRUTE_CUDA_CHECK(
            cudaMalloc(
                &d_LSE,
                static_cast<std::size_t>(B) *
                H *
                Q_LEN *
                sizeof(float)));

        // ------------------------------------------------------------
        // Deterministic initialization
        // ------------------------------------------------------------

        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<__nv_bfloat16> h_Q(q_elements);
        std::vector<__nv_bfloat16> h_K(kv_elements);
        std::vector<__nv_bfloat16> h_V(kv_elements);

        for (auto& x : h_Q)
            x = __float2bfloat16(dist(rng));

        for (auto& x : h_K)
            x = __float2bfloat16(dist(rng));

        for (auto& x : h_V)
            x = __float2bfloat16(dist(rng));

        BRUTE_CUDA_CHECK(
            cudaMemcpy(
                d_Q,
                h_Q.data(),
                q_elements * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));

        BRUTE_CUDA_CHECK(
            cudaMemcpy(
                d_K,
                h_K.data(),
                kv_elements * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));

        BRUTE_CUDA_CHECK(
            cudaMemcpy(
                d_V,
                h_V.data(),
                kv_elements * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));

        // ------------------------------------------------------------
        // Configuration search
        // ------------------------------------------------------------

        std::vector<Config> results;

        for (int num_warps : num_warps_list)
        {
            const int block_m = 16 * num_warps;

            for (int block_n : block_ns)
            {
                if (block_m > block_n)
                    continue;

                Config cfg =
                    make_config(
                        block_n,
                        D_HEAD,
                        num_warps,
                        CAUSAL);

                std::cout
                    << "Testing "
                    << "BM=" << cfg.block_m
                    << " BN=" << cfg.block_n
                    << " DH=" << cfg.d_head
                    << " NW=" << cfg.num_warps
                    << " CAUSAL=" << (cfg.causal ? 1 : 0)
                    << " SMEM="
                    << (cfg.shared_bytes / 1024.0)
                    << " KB ... ";

                bool ok =
                    benchmark_config(
                        cfg,
                        d_Q,
                        d_K,
                        d_V,
                        d_O,
                        d_LSE,
                        B,
                        H,
                        NUM_KV_HEADS,
                        Q_LEN,
                        KV_LEN,
                        KV_STRIDE,
                        scale,
                        stream,
                        WARMUP,
                        ITERATIONS);

                if (ok)
                {
                    std::cout
                        << "OK  "
                        << std::fixed
                        << std::setprecision(3)
                        << cfg.latency_ms
                        << " ms  "
                        << std::setprecision(2)
                        << cfg.tflops
                        << " TFLOP/s\n";
                }
                else
                {
                    std::cout
                        << "FAIL: "
                        << cfg.failure_reason
                        << "\n";
                }

                results.push_back(cfg);

                cudaGetLastError();
            }
        }

        // ------------------------------------------------------------
        // Sort results
        // ------------------------------------------------------------

        std::vector<Config> stable;

        for (const Config& c : results)
        {
            if (c.launched && c.numerically_valid)
            {
                stable.push_back(c);
            }
        }

        std::sort(
            stable.begin(),
            stable.end(),
            [](const Config& a, const Config& b)
            {
                return a.latency_ms < b.latency_ms;
            });

        // ------------------------------------------------------------
        // Print best configurations
        // ------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << " BEST CONFIGURATIONS: "
            << workload_name
            << " Q=" << Q_LEN
            << " KV=" << KV_LEN
            << "\n"
            << "------------------------------------------------------------\n";

        std::cout
            << std::left
            << std::setw(5)  << "Rank"
            << std::setw(6)  << "BM"
            << std::setw(6)  << "BN"
            << std::setw(6)  << "NW"
            << std::setw(12) << "Latency"
            << std::setw(12) << "TFLOP/s"
            << "\n";

        for (std::size_t i = 0;
             i < stable.size() && i < 10;
             ++i)
        {
            const Config& c = stable[i];

            std::cout
                << std::left
                << std::setw(5) << (i + 1)
                << std::setw(6) << c.block_m
                << std::setw(6) << c.block_n
                << std::setw(6) << c.num_warps
                << std::setw(12) << c.latency_ms
                << std::setw(12) << c.tflops
                << "\n";
        }

        // ------------------------------------------------------------
        // Cleanup THIS workload
        // ------------------------------------------------------------

        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_LSE);
    };

    // ================================================================
    // RUN ALL PREFILL WORKLOADS
    // ================================================================

    for (int length : prefill_lengths)
    {
        run_workload(
            length,
            length,
            "PREFILL");
    }

    // ================================================================
    // RUN ALL DECODE WORKLOADS
    // ================================================================

    for (int kv_len : decode_kv_lengths)
    {
        run_workload(
            1,
            kv_len,
            "DECODE");
    }

    // ================================================================
    // FINAL CLEANUP
    // ================================================================

    cudaStreamDestroy(stream);

    std::cout
        << "\n============================================================\n"
        << " ALL WORKLOADS COMPLETE\n"
        << "============================================================\n";

    return 0;
}