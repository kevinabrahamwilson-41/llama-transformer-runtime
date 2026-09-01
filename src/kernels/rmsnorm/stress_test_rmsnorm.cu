// stress_test_rmsnorm.cu
//
// Benchmark driver for RMSNorm.
//
// Compile:
// nvcc -arch=sm_89 -O3 stress_test_rmsnorm.cu rmsnorm_pure.cu \
//      -o stress_test_rmsnorm
//
// Run:
// ./stress_test_rmsnorm
//
// Configuration:
//   dtype        = BF16
//   hidden       = 2048
//   threads/block = 256
//   warmups      = 20
//   iterations   = 100
//
// Sequence lengths:
//   32, 64, 128, 256, 512, 1024, 2048
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

constexpr int HIDDEN_SIZE       = 2048;
constexpr int THREADS_PER_BLOCK = 256;

constexpr int WARMUP_ITERS      = 20;
constexpr int BENCH_ITERS       = 100;

constexpr float EPSILON         = 1e-5f;


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
// RMSNorm launcher declaration
// ============================================================================
//
// This is the explicit template instantiation provided by rmsnorm_pure.cu.
//

template<int HIDDEN>
void rmsnorm_launch(
    const __nv_bfloat16* d_in,
    const __nv_bfloat16* d_weight,
    __nv_bfloat16* d_out,
    int rows,
    float eps,
    int threads_per_block
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


    // Average
    const double sum =
        std::accumulate(
            sorted.begin(),
            sorted.end(),
            0.0);

    const double avg =
        sum / static_cast<double>(sorted.size());


    // Median
    double median;

    if (sorted.size() % 2 == 0) {

        const size_t mid =
            sorted.size() / 2;

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


    // Min / max
    const double min_val =
        static_cast<double>(sorted.front());

    const double max_val =
        static_cast<double>(sorted.back());


    // Standard deviation
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
        << "RMSNORM KERNEL BENCHMARK\n";

    std::cout
        << "============================================================\n";

    std::cout
        << "GPU                : "
        << prop.name
        << "\n";

    std::cout
        << "Compute Capability  : "
        << prop.major
        << "."
        << prop.minor
        << "\n";

    std::cout
        << "Global Memory       : "
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(prop.totalGlobalMem) /
               (1024.0 * 1024.0 * 1024.0)
        << " GB\n";

    std::cout
        << "SM Count            : "
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


    // ------------------------------------------------------------------------
    // RMSNorm operates on:
    //
    //     [rows, hidden]
    //
    // where:
    //
    //     rows   = sequence length
    //     hidden = 2048
    //
    // ------------------------------------------------------------------------

    const int rows = seq_len;


    const size_t input_elements =
        static_cast<size_t>(rows) *
        HIDDEN_SIZE;

    const size_t weight_elements =
        HIDDEN_SIZE;

    const size_t output_elements =
        input_elements;


    const size_t input_bytes =
        input_elements *
        sizeof(scalar_t);

    const size_t weight_bytes =
        weight_elements *
        sizeof(scalar_t);

    const size_t output_bytes =
        output_elements *
        sizeof(scalar_t);


    // ------------------------------------------------------------------------
    // Device memory
    // ------------------------------------------------------------------------

    scalar_t* d_input  = nullptr;
    scalar_t* d_weight = nullptr;
    scalar_t* d_output = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            input_bytes));

    CUDA_CHECK(
        cudaMalloc(
            &d_weight,
            weight_bytes));

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            output_bytes));


    // ------------------------------------------------------------------------
    // Initialize
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_input,
            0,
            input_bytes));

    CUDA_CHECK(
        cudaMemset(
            d_weight,
            0,
            weight_bytes));

    CUDA_CHECK(
        cudaMemset(
            d_output,
            0,
            output_bytes));


    // ------------------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------------------

    for (int i = 0;
         i < WARMUP_ITERS;
         ++i)
    {
        rmsnorm_launch<HIDDEN_SIZE>(
            d_input,
            d_weight,
            d_output,
            rows,
            EPSILON,
            THREADS_PER_BLOCK);
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

    timings.reserve(
        BENCH_ITERS);


    // ------------------------------------------------------------------------
    // Benchmark
    // ------------------------------------------------------------------------

    for (int i = 0;
         i < BENCH_ITERS;
         ++i)
    {
        CUDA_CHECK(
            cudaEventRecord(start));


        rmsnorm_launch<HIDDEN_SIZE>(
            d_input,
            d_weight,
            d_output,
            rows,
            EPSILON,
            THREADS_PER_BLOCK);


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


        // Convert milliseconds -> microseconds
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
        cudaFree(d_input));

    CUDA_CHECK(
        cudaFree(d_weight));

    CUDA_CHECK(
        cudaFree(d_output));


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
        << "Data type          : BF16\n";

    std::cout
        << "Hidden size        : "
        << HIDDEN_SIZE
        << "\n";

    std::cout
        << "Threads/block      : "
        << THREADS_PER_BLOCK
        << "\n";

    std::cout
        << "Operation          : RMSNorm\n";

    std::cout
        << "Warmup iterations  : "
        << WARMUP_ITERS
        << "\n";

    std::cout
        << "Benchmark iters    : "
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
    // Run benchmark
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