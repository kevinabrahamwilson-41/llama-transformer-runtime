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

#include "residual_add.hpp"

// ============================================================
// Configuration
// ============================================================

constexpr int HIDDEN_SIZE  = 2048;
constexpr int WARMUP_ITERS = 20;
constexpr int BENCH_ITERS  = 100;

// ============================================================
// CUDA error checking
// ============================================================

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ============================================================
// Residual-add launcher declaration
// ============================================================

namespace transformer {

void residual_add(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int64_t num_elements
);

}

// ============================================================
// Statistics
// ============================================================

struct Stats {
    double avg;
    double median;
    double min;
    double max;
    double stddev;
};

static Stats calculate_stats(const std::vector<float>& values)
{
    std::vector<float> sorted = values;

    std::sort(sorted.begin(), sorted.end());

    double sum =
        std::accumulate(
            sorted.begin(),
            sorted.end(),
            0.0
        );

    double avg =
        sum / static_cast<double>(sorted.size());

    double median;

    if (sorted.size() % 2 == 0) {
        size_t mid = sorted.size() / 2;

        median =
            (static_cast<double>(sorted[mid - 1]) +
             static_cast<double>(sorted[mid])) / 2.0;
    }
    else {
        median =
            sorted[sorted.size() / 2];
    }

    double variance = 0.0;

    for (float value : sorted) {

        double diff =
            static_cast<double>(value) - avg;

        variance += diff * diff;
    }

    variance /= static_cast<double>(sorted.size());

    return {
        avg,
        median,
        sorted.front(),
        sorted.back(),
        std::sqrt(variance)
    };
}

// ============================================================
// GPU information
// ============================================================

static void print_gpu_info()
{
    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(&prop, 0)
    );

    std::cout
        << "============================================================\n"
        << "RESIDUAL ADD KERNEL BENCHMARK\n"
        << "============================================================\n";

    std::cout
        << "GPU                : "
        << prop.name << "\n";

    std::cout
        << "Compute Capability  : "
        << prop.major << "."
        << prop.minor << "\n";

    std::cout
        << "Global Memory       : "
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(prop.totalGlobalMem)
           / (1024.0 * 1024.0 * 1024.0)
        << " GB\n";

    std::cout
        << "SM Count            : "
        << prop.multiProcessorCount
        << "\n";

    std::cout
        << "============================================================\n\n";
}

// ============================================================
// Benchmark one sequence length
// ============================================================

static Stats benchmark_sequence_length(int seq_len)
{
    using scalar_t = __nv_bfloat16;

    // --------------------------------------------------------
    // Tensor size
    //
    // [seq_len, hidden_size]
    // --------------------------------------------------------

    const size_t elements =
        static_cast<size_t>(seq_len)
        * HIDDEN_SIZE;

    const size_t bytes =
        elements * sizeof(scalar_t);

    // --------------------------------------------------------
    // Device allocations
    // --------------------------------------------------------

    scalar_t* d_a   = nullptr;
    scalar_t* d_b   = nullptr;
    scalar_t* d_out = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_a,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_b,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_out,
            bytes
        )
    );

    // --------------------------------------------------------
    // Initialize
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_a,
            0,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_b,
            0,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_out,
            0,
            bytes
        )
    );

    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for (int i = 0; i < WARMUP_ITERS; ++i) {

        transformer::residual_add(
            d_a,
            d_b,
            d_out,
            static_cast<int64_t>(elements)
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );

    std::vector<float> timings;
    timings.reserve(BENCH_ITERS);

    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    for (int i = 0; i < BENCH_ITERS; ++i) {

        CUDA_CHECK(
            cudaEventRecord(start)
        );

        transformer::residual_add(
            d_a,
            d_b,
            d_out,
            static_cast<int64_t>(elements)
        );

        CUDA_CHECK(
            cudaEventRecord(stop)
        );

        CUDA_CHECK(
            cudaEventSynchronize(stop)
        );

        float elapsed_ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &elapsed_ms,
                start,
                stop
            )
        );

        timings.push_back(
            elapsed_ms * 1000.0f
        );
    }

    Stats stats =
        calculate_stats(timings);

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaEventDestroy(start)
    );

    CUDA_CHECK(
        cudaEventDestroy(stop)
    );

    CUDA_CHECK(
        cudaFree(d_a)
    );

    CUDA_CHECK(
        cudaFree(d_b)
    );

    CUDA_CHECK(
        cudaFree(d_out)
    );

    return stats;
}

// ============================================================
// Main
// ============================================================

int main()
{
    CUDA_CHECK(
        cudaSetDevice(0)
    );

    print_gpu_info();

    std::cout
        << "Configuration\n"
        << "-------------\n";

    std::cout
        << "Data type          : BF16\n";

    std::cout
        << "Hidden size        : "
        << HIDDEN_SIZE
        << "\n";

    std::cout
        << "Operation          : out = a + b\n";

    std::cout
        << "Threads/block      : 128\n";

    std::cout
        << "Warmup iterations  : "
        << WARMUP_ITERS
        << "\n";

    std::cout
        << "Benchmark iters    : "
        << BENCH_ITERS
        << "\n\n";

    // ========================================================
    // Required sequence lengths
    // ========================================================

    const int sequence_lengths[] = {
        32,
        64,
        128,
        256,
        512,
        1024,
        2048
    };

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

    for (int seq_len : sequence_lengths) {

        std::cout
            << std::left
            << std::setw(12)
            << seq_len
            << std::flush;

        Stats stats =
            benchmark_sequence_length(seq_len);

        std::cout
            << std::fixed
            << std::setprecision(3)
            << std::setw(16) << stats.avg
            << std::setw(16) << stats.median
            << std::setw(16) << stats.min
            << std::setw(16) << stats.max
            << std::setw(16) << stats.stddev
            << "\n";
    }

    std::cout
        << "\n============================================================\n"
        << "Benchmark complete\n"
        << "============================================================\n";

    CUDA_CHECK(
        cudaDeviceReset()
    );

    return 0;
}
