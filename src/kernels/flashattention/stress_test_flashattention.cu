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
//   BM           = 64
//   BN           = 64
//   warps        = 4
//   warmups      = 20
//   iterations   = 100
//
// Output:
//   Average latency
//   Median latency
//   TFLOP/s
//   Effective bandwidth
//
// NOTE:
//   This benchmark intentionally uses a fixed kernel configuration.
//   BM=64, BN=64, 4 warps are the pre-optimization baseline.
//
//   The reported bandwidth is an ALGORITHMIC EFFECTIVE BANDWIDTH metric
//   based on tensor I/O bytes, NOT measured DRAM bandwidth.
//
// ============================================================================

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

static constexpr int BATCH_SIZE   = 1;

static constexpr int NUM_Q_HEADS  = 32;
static constexpr int NUM_KV_HEADS = 8;

static constexpr int D_HEAD       = 64;

// Fixed baseline kernel configuration.
// These values come directly from launch_fat_variant() in flashattention.cu.
static constexpr int BLOCK_M      = 64;
static constexpr int BLOCK_N      = 64;
static constexpr int NUM_WARPS    = 4;

static constexpr int WARMUP_ITERS = 20;
static constexpr int BENCH_ITERS  = 100;

static constexpr bool CAUSAL      = true;


// ============================================================================
// Benchmark result
// ============================================================================

struct BenchmarkResult {

    double avg_us;
    double median_us;

    double tflops;

    double effective_bw;
};


// ============================================================================
// Statistics
// ============================================================================

static double calculate_average(
    const std::vector<float>& values)
{
    if (values.empty())
        return 0.0;

    const double sum =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0);

    return sum /
           static_cast<double>(values.size());
}


static double calculate_median(
    std::vector<float> values)
{
    if (values.empty())
        return 0.0;

    std::sort(
        values.begin(),
        values.end());

    const size_t n = values.size();

    if (n % 2 == 0) {

        return (
            static_cast<double>(values[n / 2 - 1]) +
            static_cast<double>(values[n / 2])
        ) / 2.0;
    }

    return static_cast<double>(values[n / 2]);
}


// ============================================================================
// FLOP calculation
// ============================================================================
//
// Attention consists primarily of:
//
//     S = Q K^T
//     O = softmax(S) V
//
// Each matrix multiplication contributes approximately:
//
//     2 * Q_LEN * KV_LEN * D_HEAD
//
// FLOPs.
//
// Therefore:
//
//     4 * B * Hq * Q_LEN * KV_LEN * D_HEAD
//
// This is the standard dense-attention FLOP accounting.
//
// For causal attention, masked elements are not conceptually part of the
// useful attention computation. However, the kernel still processes tiles
// containing masked positions.
//
// Therefore this metric is intentionally reported as a standard
// "Attention TFLOP/s" metric rather than claiming exact hardware
// instruction FLOPs.
// ============================================================================

static double calculate_attention_tflops(
    int q_seq_len,
    int kv_seq_len,
    double latency_us)
{
    const double flops =
        4.0 *
        static_cast<double>(BATCH_SIZE) *
        static_cast<double>(NUM_Q_HEADS) *
        static_cast<double>(q_seq_len) *
        static_cast<double>(kv_seq_len) *
        static_cast<double>(D_HEAD);

    const double seconds =
        latency_us * 1.0e-6;

    return (flops / seconds) / 1.0e12;
}


// ============================================================================
// Effective bandwidth
// ============================================================================
//
// This is an algorithmic I/O bandwidth metric.
//
// Counted tensors:
//
//     Q read
//     K read
//     V read
//     O write
//     LSE write
//
// BF16 = 2 bytes
// FP32 LSE = 4 bytes
//
// IMPORTANT:
// This is NOT Nsight's measured DRAM throughput.
//
// FlashAttention reuses K/V through shared memory and therefore the actual
// global-memory traffic depends on the kernel tiling and number of Q blocks.
// For the presentation, call this "Effective BW" and document the
// methodology.
// ============================================================================

static double calculate_effective_bandwidth_gbps(
    int q_seq_len,
    int kv_seq_len,
    double latency_us)
{
    const double q_bytes =
        static_cast<double>(BATCH_SIZE) *
        NUM_Q_HEADS *
        q_seq_len *
        D_HEAD *
        sizeof(__nv_bfloat16);

    const double kv_bytes =
        static_cast<double>(BATCH_SIZE) *
        NUM_KV_HEADS *
        kv_seq_len *
        D_HEAD *
        sizeof(__nv_bfloat16);

    const double v_bytes = kv_bytes;

    const double o_bytes =
        static_cast<double>(BATCH_SIZE) *
        NUM_Q_HEADS *
        q_seq_len *
        D_HEAD *
        sizeof(__nv_bfloat16);

    const double lse_bytes =
        static_cast<double>(BATCH_SIZE) *
        NUM_Q_HEADS *
        q_seq_len *
        sizeof(float);

    const double total_bytes =
        q_bytes +
        kv_bytes +
        v_bytes +
        o_bytes +
        lse_bytes;

    const double seconds =
        latency_us * 1.0e-6;

    return (
        total_bytes /
        seconds
    ) / 1.0e9;
}


// ============================================================================
// GPU information
// ============================================================================

static void print_gpu_info()
{
    cudaDeviceProp prop{};

    CHECK_CUDA(
        cudaGetDeviceProperties(
            &prop,
            0));

    std::cout
        << "============================================================\n";

    std::cout
        << "FLASHATTENTION BASELINE BENCHMARK\n";

    std::cout
        << "============================================================\n";

    std::cout
        << "GPU               : "
        << prop.name
        << "\n";

    std::cout
        << "Compute Capability : "
        << prop.major
        << "."
        << prop.minor
        << "\n";

    std::cout
        << "Global Memory      : "
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(
               prop.totalGlobalMem)
           / (1024.0 * 1024.0 * 1024.0)
        << " GB\n";

    std::cout
        << "SM Count           : "
        << prop.multiProcessorCount
        << "\n";

    std::cout
        << "============================================================\n\n";
}


// ============================================================================
// Benchmark one sequence length
// ============================================================================

static BenchmarkResult benchmark_sequence_length(
    int seq_len)
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

    const size_t o_elements =
        q_elements;

    const size_t lse_elements =
        static_cast<size_t>(BATCH_SIZE) *
        NUM_Q_HEADS *
        q_seq_len;


    // ------------------------------------------------------------------------
    // Device allocation
    // ------------------------------------------------------------------------

    scalar_t* d_Q = nullptr;
    scalar_t* d_K = nullptr;
    scalar_t* d_V = nullptr;
    scalar_t* d_O = nullptr;

    float* d_L = nullptr;


    CHECK_CUDA(
        cudaMalloc(
            &d_Q,
            q_elements * sizeof(scalar_t)));

    CHECK_CUDA(
        cudaMalloc(
            &d_K,
            kv_elements * sizeof(scalar_t)));

    CHECK_CUDA(
        cudaMalloc(
            &d_V,
            kv_elements * sizeof(scalar_t)));

    CHECK_CUDA(
        cudaMalloc(
            &d_O,
            o_elements * sizeof(scalar_t)));

    CHECK_CUDA(
        cudaMalloc(
            &d_L,
            lse_elements * sizeof(float)));


    // ------------------------------------------------------------------------
    // Initialization
    //
    // Zero initialization is intentional for the latency baseline.
    //
    // Correctness/error validation should be performed separately using
    // deterministic non-zero input.
    // ------------------------------------------------------------------------

    CHECK_CUDA(
        cudaMemset(
            d_Q,
            0,
            q_elements * sizeof(scalar_t)));

    CHECK_CUDA(
        cudaMemset(
            d_K,
            0,
            kv_elements * sizeof(scalar_t)));

    CHECK_CUDA(
        cudaMemset(
            d_V,
            0,
            kv_elements * sizeof(scalar_t)));

    CHECK_CUDA(
        cudaMemset(
            d_O,
            0,
            o_elements * sizeof(scalar_t)));

    CHECK_CUDA(
        cudaMemset(
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

    params.batch_size =
        BATCH_SIZE;

    params.num_heads =
        NUM_Q_HEADS;

    params.num_kv_heads =
        NUM_KV_HEADS;

    params.q_seq_len =
        q_seq_len;

    params.kv_seq_len =
        kv_seq_len;

    params.d_head =
        D_HEAD;


    // For contiguous:
    //
    // [B, Hkv, S, D]
    //
    // the stride between KV heads is S * D elements.
    //
    params.kv_stride =
        kv_seq_len;


    params.scale =
        1.0f /
        std::sqrt(
            static_cast<float>(D_HEAD));

    params.causal =
        CAUSAL;

    params.dtype =
        transformer::DType::BF16;

    params.stream =
        nullptr;


    // ------------------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------------------

    for (int i = 0;
         i < WARMUP_ITERS;
         ++i)
    {
        transformer::launch_flash_attention(
            params);
    }

    CHECK_CUDA(
        cudaDeviceSynchronize());


    // ------------------------------------------------------------------------
    // Benchmark
    // ------------------------------------------------------------------------

    std::vector<float> timings;

    timings.reserve(
        BENCH_ITERS);


    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(
        cudaEventCreate(
            &start));

    CHECK_CUDA(
        cudaEventCreate(
            &stop));


    for (int i = 0;
         i < BENCH_ITERS;
         ++i)
    {
        CHECK_CUDA(
            cudaEventRecord(
                start,
                params.stream));


        transformer::launch_flash_attention(
            params);


        CHECK_CUDA(
            cudaEventRecord(
                stop,
                params.stream));


        CHECK_CUDA(
            cudaEventSynchronize(
                stop));


        float elapsed_ms = 0.0f;

        CHECK_CUDA(
            cudaEventElapsedTime(
                &elapsed_ms,
                start,
                stop));


        timings.push_back(
            elapsed_ms * 1000.0f);
    }


    CHECK_CUDA(
        cudaEventDestroy(
            start));

    CHECK_CUDA(
        cudaEventDestroy(
            stop));


    // ------------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------------

    const double avg_us =
        calculate_average(
            timings);

    const double median_us =
        calculate_median(
            timings);


    // ------------------------------------------------------------------------
    // Performance metrics
    // ------------------------------------------------------------------------

    const double tflops =
        calculate_attention_tflops(
            q_seq_len,
            kv_seq_len,
            avg_us);


    const double effective_bw =
        calculate_effective_bandwidth_gbps(
            q_seq_len,
            kv_seq_len,
            avg_us);


    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------

    CHECK_CUDA(
        cudaFree(d_Q));

    CHECK_CUDA(
        cudaFree(d_K));

    CHECK_CUDA(
        cudaFree(d_V));

    CHECK_CUDA(
        cudaFree(d_O));

    CHECK_CUDA(
        cudaFree(d_L));


    return {
        avg_us,
        median_us,
        tflops,
        effective_bw
    };
}


// ============================================================================
// Main
// ============================================================================

int main()
{
    CHECK_CUDA(
        cudaSetDevice(0));


    print_gpu_info();


    // ------------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------------

    std::cout
        << "Configuration\n";

    std::cout
        << "-------------\n";

    std::cout
        << "Data type         : BF16\n";

    std::cout
        << "Batch size        : "
        << BATCH_SIZE
        << "\n";

    std::cout
        << "Q heads           : "
        << NUM_Q_HEADS
        << "\n";

    std::cout
        << "KV heads          : "
        << NUM_KV_HEADS
        << "\n";

    std::cout
        << "Head dimension    : "
        << D_HEAD
        << "\n";

    std::cout
        << "Causal            : "
        << (CAUSAL ? "true" : "false")
        << "\n";

    std::cout
        << "Block M           : "
        << BLOCK_M
        << "\n";

    std::cout
        << "Block N           : "
        << BLOCK_N
        << "\n";

    std::cout
        << "Warps              : "
        << NUM_WARPS
        << "\n";

    std::cout
        << "Warmup iterations : "
        << WARMUP_ITERS
        << "\n";

    std::cout
        << "Benchmark iters   : "
        << BENCH_ITERS
        << "\n\n";


    // ------------------------------------------------------------------------
    // Sequence lengths
    // ------------------------------------------------------------------------

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
    // Output header
    // ------------------------------------------------------------------------

    std::cout
        << std::left
        << std::setw(10)
        << "Seq Len"

        << std::setw(16)
        << "Avg us"

        << std::setw(16)
        << "Median us"

        << std::setw(16)
        << "TFLOP/s"

        << std::setw(20)
        << "Effective BW"

        << "\n";


    std::cout
        << std::string(
               78,
               '-')
        << "\n";


    // ------------------------------------------------------------------------
    // Benchmark
    // ------------------------------------------------------------------------

    for (int i = 0;
         i < NUM_LENGTHS;
         ++i)
    {
        const int seq_len =
            sequence_lengths[i];


        std::cout
            << std::left
            << std::setw(10)
            << seq_len
            << std::flush;


        BenchmarkResult result =
            benchmark_sequence_length(
                seq_len);


        std::cout
            << std::fixed
            << std::setprecision(3)

            << std::setw(16)
            << result.avg_us

            << std::setw(16)
            << result.median_us

            << std::setw(16)
            << result.tflops

            << std::setw(20)
            << result.effective_bw

            << "\n";
    }


    std::cout
        << "\n";

    std::cout
        << "============================================================\n";

    std::cout
        << "Benchmark complete\n";

    std::cout
        << "============================================================\n";


    CHECK_CUDA(
        cudaDeviceReset());


    return 0;
}