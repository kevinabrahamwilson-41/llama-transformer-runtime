// stress_test_flashattention.cu
//
// Benchmark driver for flashattention.cu
//
// Compile:
// nvcc -arch=sm_89 -O3 stress_test_flashattention.cu flashattention.cu \
//      -o stress_test_flashattention
//
// Run:
// ./stress_test_flashattention
//
// Benchmarks:
// Sequence lengths: 32, 64, 128, 256, 512, 1024, 2048
//
// Configuration:
//   dtype        = BF16
//   batch        = 1
//   Q heads      = 32
//   KV heads     = 8
//   head dim     = 64
//   causal       = true
//   warmups      = 20
//   iterations   = 100
//
// Output latency is CUDA kernel execution time in microseconds.
//

#include "flashattention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr,                                                 \
                    "CUDA error at %s:%d: %s\n",                             \
                    __FILE__,                                                \
                    __LINE__,                                                \
                    cudaGetErrorString(err__));                              \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


// ============================================================================
// Benchmark configuration
// ============================================================================

static constexpr int BATCH_SIZE    = 1;
static constexpr int NUM_Q_HEADS   = 32;
static constexpr int NUM_KV_HEADS  = 8;
static constexpr int D_HEAD        = 64;

static constexpr int WARMUP_ITERS  = 20;
static constexpr int BENCH_ITERS   = 100;

static constexpr bool CAUSAL = true;


// ============================================================================
// Statistics
// ============================================================================

struct Stats {
    double avg;
    double median;
    double min;
    double max;
    double stddev;
};


static Stats calculate_stats(const std::vector<float>& values)
{
    if (values.empty()) {
        return {0.0, 0.0, 0.0, 0.0, 0.0};
    }

    std::vector<float> sorted = values;

    std::sort(sorted.begin(), sorted.end());

    const double sum =
        std::accumulate(sorted.begin(), sorted.end(), 0.0);

    const double avg =
        sum / static_cast<double>(sorted.size());

    double median;

    if (sorted.size() % 2 == 0) {
        const size_t mid = sorted.size() / 2;

        median =
            (static_cast<double>(sorted[mid - 1]) +
             static_cast<double>(sorted[mid])) / 2.0;
    }
    else {
        median =
            static_cast<double>(sorted[sorted.size() / 2]);
    }

    const double min_val =
        static_cast<double>(sorted.front());

    const double max_val =
        static_cast<double>(sorted.back());

    double variance = 0.0;

    for (float v : sorted) {
        const double diff =
            static_cast<double>(v) - avg;

        variance += diff * diff;
    }

    variance /= static_cast<double>(sorted.size());

    const double stddev = std::sqrt(variance);

    return {
        avg,
        median,
        min_val,
        max_val,
        stddev
    };
}


// ============================================================================
// Device initialization
// ============================================================================

static void print_gpu_info()
{
    cudaDeviceProp prop{};

    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << "============================================================\n";
    std::cout << "FLASHATTENTION KERNEL BENCHMARK\n";
    std::cout << "============================================================\n";

    std::cout << "GPU              : " << prop.name << "\n";
    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << "\n";

    std::cout << "Global Memory     : "
              << static_cast<double>(prop.totalGlobalMem) /
                     (1024.0 * 1024.0 * 1024.0)
              << " GB\n";

    std::cout << "SM Count          : "
              << prop.multiProcessorCount << "\n";

    std::cout << "============================================================\n\n";
}


// ============================================================================
// Benchmark one sequence length
// ============================================================================

static Stats benchmark_sequence_length(int seq_len)
{
    using scalar_t = __nv_bfloat16;

    const int q_seq_len  = seq_len;
    const int kv_seq_len = seq_len;

    // ------------------------------------------------------------------------
    // Tensor sizes
    //
    // Q:
    //   [B, Hq, S, D]
    //
    // K/V:
    //   [B, Hkv, S, D]
    //
    // O:
    //   [B, Hq, S, D]
    //
    // LSE:
    //   [B, Hq, S]
    // ------------------------------------------------------------------------

    const size_t q_elements =
        static_cast<size_t>(BATCH_SIZE) *
        NUM_Q_HEADS *
        q_seq_len *
        D_HEAD;

    const size_t kv_elements =
        static_cast<size_t>(BATCH_SIZE) *
        NUM_KV_HEADS *
        kv_seq_len *
        D_HEAD;

    const size_t o_elements = q_elements;

    const size_t lse_elements =
        static_cast<size_t>(BATCH_SIZE) *
        NUM_Q_HEADS *
        q_seq_len;

    // ------------------------------------------------------------------------
    // Allocate device memory
    // ------------------------------------------------------------------------

    scalar_t* d_Q  = nullptr;
    scalar_t* d_K  = nullptr;
    scalar_t* d_V  = nullptr;
    scalar_t* d_O  = nullptr;
    float*    d_L  = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_Q,
        q_elements * sizeof(scalar_t)));

    CHECK_CUDA(cudaMalloc(
        &d_K,
        kv_elements * sizeof(scalar_t)));

    CHECK_CUDA(cudaMalloc(
        &d_V,
        kv_elements * sizeof(scalar_t)));

    CHECK_CUDA(cudaMalloc(
        &d_O,
        o_elements * sizeof(scalar_t)));

    CHECK_CUDA(cudaMalloc(
        &d_L,
        lse_elements * sizeof(float)));

    // ------------------------------------------------------------------------
    // Initialize inputs
    //
    // We don't need meaningful values for a pure latency benchmark.
    // Using cudaMemset gives deterministic initialization.
    // ------------------------------------------------------------------------

    CHECK_CUDA(cudaMemset(
        d_Q,
        0,
        q_elements * sizeof(scalar_t)));

    CHECK_CUDA(cudaMemset(
        d_K,
        0,
        kv_elements * sizeof(scalar_t)));

    CHECK_CUDA(cudaMemset(
        d_V,
        0,
        kv_elements * sizeof(scalar_t)));

    CHECK_CUDA(cudaMemset(
        d_O,
        0,
        o_elements * sizeof(scalar_t)));

    CHECK_CUDA(cudaMemset(
        d_L,
        0,
        lse_elements * sizeof(float)));

    // ------------------------------------------------------------------------
    // FlashAttention parameters
    // ------------------------------------------------------------------------

    transformer::FlashAttentionParams params{};

    params.Q = d_Q;
    params.K = d_K;
    params.V = d_V;
    params.O = d_O;
    params.L = d_L;

    params.batch_size    = BATCH_SIZE;
    params.num_heads     = NUM_Q_HEADS;
    params.num_kv_heads  = NUM_KV_HEADS;

    params.q_seq_len     = q_seq_len;
    params.kv_seq_len    = kv_seq_len;

    params.d_head        = D_HEAD;

    //
    // Physical token stride between KV heads.
    //
    // Your kernel computes:
    //
    //   kv_off =
    //       (b * num_kv_heads + h_kv)
    //       * kv_stride
    //       * D_HEAD;
    //
    // Therefore for contiguous [B,Hkv,S,D]:
    //
    params.kv_stride = kv_seq_len;

    params.scale =
        1.0f / std::sqrt(static_cast<float>(D_HEAD));

    params.causal = CAUSAL;

    params.dtype = transformer::DType::BF16;

    params.stream = nullptr;

    // ------------------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------------------

    for (int i = 0; i < WARMUP_ITERS; ++i) {

        transformer::launch_flash_attention(params);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // ------------------------------------------------------------------------
    // Benchmark
    //
    // Each iteration gets its own CUDA events.
    // This measures GPU execution time rather than CPU launch overhead.
    // ------------------------------------------------------------------------

    std::vector<float> timings;
    timings.reserve(BENCH_ITERS);

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < BENCH_ITERS; ++i) {

        CHECK_CUDA(cudaEventRecord(start, params.stream));

        transformer::launch_flash_attention(params);

        CHECK_CUDA(cudaEventRecord(stop, params.stream));

        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop));

        timings.push_back(elapsed_ms * 1000.0f);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // ------------------------------------------------------------------------
    // Calculate statistics
    // ------------------------------------------------------------------------

    Stats stats = calculate_stats(timings);

    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_L));

    return stats;
}


// ============================================================================
// Main
// ============================================================================

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    print_gpu_info();

    std::cout << "Configuration\n";
    std::cout << "-------------\n";

    std::cout << "Data type         : BF16\n";
    std::cout << "Batch size        : " << BATCH_SIZE << "\n";
    std::cout << "Q heads           : " << NUM_Q_HEADS << "\n";
    std::cout << "KV heads          : " << NUM_KV_HEADS << "\n";
    std::cout << "Head dimension    : " << D_HEAD << "\n";
    std::cout << "Causal            : "
              << (CAUSAL ? "true" : "false") << "\n";

    std::cout << "Warmup iterations : "
              << WARMUP_ITERS << "\n";

    std::cout << "Benchmark iters   : "
              << BENCH_ITERS << "\n\n";

    const int sequence_lengths[] = {
        32,
        64,
        128,
        256,
        512,
        1024,
        2048
    };

    constexpr int NUM_LENGTHS =
        sizeof(sequence_lengths) /
        sizeof(sequence_lengths[0]);

    // ------------------------------------------------------------------------
    // Header
    // ------------------------------------------------------------------------

    std::cout << std::left
              << std::setw(12) << "Seq Len"
              << std::setw(16) << "Avg us"
              << std::setw(16) << "Median us"
              << std::setw(16) << "Min us"
              << std::setw(16) << "Max us"
              << std::setw(16) << "Std us"
              << "\n";

    std::cout << std::string(92, '-')
              << "\n";

    // ------------------------------------------------------------------------
    // Run benchmark
    // ------------------------------------------------------------------------

    for (int i = 0; i < NUM_LENGTHS; ++i) {

        const int seq_len = sequence_lengths[i];

        std::cout << std::left
                  << std::setw(12)
                  << seq_len
                  << std::flush;

        Stats stats =
            benchmark_sequence_length(seq_len);

        std::cout << std::fixed
                  << std::setprecision(3)
                  << std::setw(16)
                  << stats.avg
                  << std::setw(16)
                  << stats.median
                  << std::setw(16)
                  << stats.min
                  << std::setw(16)
                  << stats.max
                  << std::setw(16)
                  << stats.stddev
                  << "\n";
    }

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "Benchmark complete\n";
    std::cout << "============================================================\n";

    CHECK_CUDA(cudaDeviceReset());

    return 0;
}