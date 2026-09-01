// stress_test_silu_mul.cu
//
// Benchmark driver for SiLU + multiplication CUDA kernel.
//
// Compile:
// nvcc -arch=sm_89 -O3 stress_test_silu.cu silu.cu \
//      -o stress_test_silu_mul
//
// Run:
// ./stress_test_silu_mul
//
// Configuration:
//   dtype        = BF16
//   intermediate = 8192
//   warmups      = 20
//   iterations   = 100
//
// Benchmarks:
//   Sequence lengths: 32, 64, 128, 256, 512, 1024, 2048
//
// Operation:
//   out = SiLU(gate) * up
//

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>


// ============================================================================
// Configuration
// ============================================================================

constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int WARMUP_ITERS     = 20;
constexpr int BENCH_ITERS      = 100;


// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr,                                                   \
                "CUDA Error %s:%d : %s\n",                                \
                __FILE__,                                                 \
                __LINE__,                                                 \
                cudaGetErrorString(err));                                 \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                      \
} while (0)


// ============================================================================
// Launcher declaration
// ============================================================================
//
// Defined in your SiLU CUDA source file.
//

void launch_silu_mul(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    int seq_len,
    int intermediate
);


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


static Stats calculate_stats(
    const std::vector<float>& values)
{
    if (values.empty()) {
        return {0.0, 0.0, 0.0, 0.0, 0.0};
    }

    std::vector<float> sorted = values;

    std::sort(sorted.begin(), sorted.end());

    const double sum =
        std::accumulate(
            sorted.begin(),
            sorted.end(),
            0.0);

    const double avg =
        sum / static_cast<double>(sorted.size());

    double median;

    if (sorted.size() % 2 == 0) {

        const size_t mid = sorted.size() / 2;

        median =
            (static_cast<double>(sorted[mid - 1]) +
             static_cast<double>(sorted[mid])) /
            2.0;
    }
    else {

        median =
            static_cast<double>(
                sorted[sorted.size() / 2]);
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

    variance /=
        static_cast<double>(sorted.size());

    const double stddev =
        std::sqrt(variance);

    return {
        avg,
        median,
        min_val,
        max_val,
        stddev
    };
}


// ============================================================================
// GPU information
// ============================================================================

static void print_gpu_info()
{
    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(&prop, 0));

    std::cout
        << "============================================================\n";

    std::cout
        << "SILU + MUL KERNEL BENCHMARK\n";

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
        << static_cast<double>(prop.totalGlobalMem) /
               (1024.0 * 1024.0 * 1024.0)
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

static Stats benchmark_sequence_length(
    int seq_len)
{
    using scalar_t = __nv_bfloat16;

    const size_t elements =
        static_cast<size_t>(seq_len) *
        INTERMEDIATE_SIZE;

    const size_t bytes =
        elements * sizeof(scalar_t);

    scalar_t* d_gate = nullptr;
    scalar_t* d_up   = nullptr;
    scalar_t* d_out  = nullptr;


    // ------------------------------------------------------------------------
    // Allocate
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaMalloc(
            &d_gate,
            bytes));

    CUDA_CHECK(
        cudaMalloc(
            &d_up,
            bytes));

    CUDA_CHECK(
        cudaMalloc(
            &d_out,
            bytes));


    // ------------------------------------------------------------------------
    // Initialize
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_gate,
            0,
            bytes));

    CUDA_CHECK(
        cudaMemset(
            d_up,
            0,
            bytes));

    CUDA_CHECK(
        cudaMemset(
            d_out,
            0,
            bytes));


    // ------------------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------------------

    for (int i = 0; i < WARMUP_ITERS; ++i) {

        launch_silu_mul(
            d_gate,
            d_up,
            d_out,
            seq_len,
            INTERMEDIATE_SIZE);
    }

    CUDA_CHECK(
        cudaDeviceSynchronize());


    // ------------------------------------------------------------------------
    // CUDA events
    // ------------------------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start));

    CUDA_CHECK(
        cudaEventCreate(&stop));


    std::vector<float> timings;
    timings.reserve(BENCH_ITERS);


    // ------------------------------------------------------------------------
    // Benchmark
    // ------------------------------------------------------------------------

    for (int i = 0; i < BENCH_ITERS; ++i) {

        CUDA_CHECK(
            cudaEventRecord(start));

        launch_silu_mul(
            d_gate,
            d_up,
            d_out,
            seq_len,
            INTERMEDIATE_SIZE);

        CUDA_CHECK(
            cudaEventRecord(stop));

        CUDA_CHECK(
            cudaEventSynchronize(stop));


        float elapsed_ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &elapsed_ms,
                start,
                stop));


        // ms -> us
        timings.push_back(
            elapsed_ms * 1000.0f);
    }


    // ------------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------------

    Stats stats =
        calculate_stats(timings);


    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaEventDestroy(start));

    CUDA_CHECK(
        cudaEventDestroy(stop));

    CUDA_CHECK(
        cudaFree(d_gate));

    CUDA_CHECK(
        cudaFree(d_up));

    CUDA_CHECK(
        cudaFree(d_out));


    return stats;
}


// ============================================================================
// Main
// ============================================================================

int main()
{
    CUDA_CHECK(
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
        << "Intermediate size : "
        << INTERMEDIATE_SIZE
        << "\n";

    std::cout
        << "Operation         : "
        << "out = SiLU(gate) * up\n";

    std::cout
        << "Warmup iterations : "
        << WARMUP_ITERS
        << "\n";

    std::cout
        << "Benchmark iters   : "
        << BENCH_ITERS
        << "\n\n";


    // ------------------------------------------------------------------------
    // Required sequence lengths
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
    // Table header
    // ------------------------------------------------------------------------

    std::cout
        << std::left
        << std::setw(12) << "Seq Len"
        << std::setw(16) << "Avg us"
        << std::setw(16) << "Median us"
        << std::setw(16) << "Min us"
        << std::setw(16) << "Max us"
        << std::setw(16) << "Std us"
        << "\n";

    std::cout
        << std::string(92, '-')
        << "\n";


    // ------------------------------------------------------------------------
    // Run all benchmarks
    // ------------------------------------------------------------------------

    for (int i = 0;
         i < NUM_LENGTHS;
         ++i)
    {
        const int seq_len =
            sequence_lengths[i];


        std::cout
            << std::left
            << std::setw(12)
            << seq_len
            << std::flush;


        Stats stats =
            benchmark_sequence_length(
                seq_len);


        std::cout
            << std::fixed
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


    std::cout
        << "\n============================================================\n";

    std::cout
        << "Benchmark complete\n";

    std::cout
        << "============================================================\n";


    CUDA_CHECK(
        cudaDeviceReset());

    return 0;
}