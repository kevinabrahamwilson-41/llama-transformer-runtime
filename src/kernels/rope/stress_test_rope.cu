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

#include "rope.hpp"

// ============================================================
// Configuration
// ============================================================

constexpr int WARMUP_ITERS = 20;
constexpr int BENCH_ITERS  = 100;

constexpr int HEADS    = ROPE_HEADS;
constexpr int KV_HEADS = ROPE_KV_HEADS;
constexpr int HEAD_DIM = ROPE_HEAD_DIM;
constexpr int ROTARY_DIM = ROPE_ROTARY_DIM;

// ============================================================
// Launcher declaration
// ============================================================

void launch_rope_qkv(
    const __nv_bfloat16* q_in,
    __nv_bfloat16* q_out,
    const __nv_bfloat16* k_in,
    __nv_bfloat16* k_out,
    const __nv_bfloat16* v_in,
    __nv_bfloat16* v_out,
    float* cos_table,
    float* sin_table,
    int tokens,
    int position_offset
);

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

    double avg = sum / sorted.size();

    double median;

    if (sorted.size() % 2 == 0) {
        size_t mid = sorted.size() / 2;

        median =
            (sorted[mid - 1] + sorted[mid]) / 2.0;
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

    variance /= sorted.size();

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
        << "ROPE + V TRANSPOSE KERNEL BENCHMARK\n"
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
    // Tensor sizes
    // --------------------------------------------------------

    const size_t q_elements =
        static_cast<size_t>(seq_len)
        * HEADS
        * HEAD_DIM;

    const size_t kv_elements =
        static_cast<size_t>(seq_len)
        * KV_HEADS
        * HEAD_DIM;

    const size_t q_bytes =
        q_elements * sizeof(scalar_t);

    const size_t kv_bytes =
        kv_elements * sizeof(scalar_t);

    // RoPE tables:
    //
    // [position, rotary_dim]
    //
    const size_t rope_elements =
        static_cast<size_t>(seq_len)
        * ROTARY_DIM;

    const size_t rope_bytes =
        rope_elements * sizeof(float);

    // --------------------------------------------------------
    // Device allocations
    // --------------------------------------------------------

    scalar_t* d_q_in  = nullptr;
    scalar_t* d_q_out = nullptr;

    scalar_t* d_k_in  = nullptr;
    scalar_t* d_k_out = nullptr;

    scalar_t* d_v_in  = nullptr;
    scalar_t* d_v_out = nullptr;

    float* d_cos = nullptr;
    float* d_sin = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_q_in,
            q_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_q_out,
            q_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_k_in,
            kv_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_k_out,
            kv_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_v_in,
            kv_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_v_out,
            kv_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_cos,
            rope_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_sin,
            rope_bytes
        )
    );

    // --------------------------------------------------------
    // Initialize
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_q_in,
            0,
            q_bytes
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_k_in,
            0,
            kv_bytes
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_v_in,
            0,
            kv_bytes
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_q_out,
            0,
            q_bytes
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_k_out,
            0,
            kv_bytes
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_v_out,
            0,
            kv_bytes
        )
    );

    // --------------------------------------------------------
    // Fill RoPE tables with valid values
    //
    // cos = 1
    // sin = 0
    //
    // This makes the transformation numerically stable while
    // still exercising the complete kernel.
    // --------------------------------------------------------

    std::vector<float> h_cos(
        rope_elements,
        1.0f
    );

    std::vector<float> h_sin(
        rope_elements,
        0.0f
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_cos,
            h_cos.data(),
            rope_bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_sin,
            h_sin.data(),
            rope_bytes,
            cudaMemcpyHostToDevice
        )
    );

    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for (int i = 0; i < WARMUP_ITERS; ++i) {

        launch_rope_qkv(
            d_q_in,
            d_q_out,
            d_k_in,
            d_k_out,
            d_v_in,
            d_v_out,
            d_cos,
            d_sin,
            seq_len,
            0
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    // --------------------------------------------------------
    // Benchmark
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

    for (int i = 0; i < BENCH_ITERS; ++i) {

        CUDA_CHECK(
            cudaEventRecord(start)
        );

        launch_rope_qkv(
            d_q_in,
            d_q_out,
            d_k_in,
            d_k_out,
            d_v_in,
            d_v_out,
            d_cos,
            d_sin,
            seq_len,
            0
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
        cudaFree(d_q_in)
    );

    CUDA_CHECK(
        cudaFree(d_q_out)
    );

    CUDA_CHECK(
        cudaFree(d_k_in)
    );

    CUDA_CHECK(
        cudaFree(d_k_out)
    );

    CUDA_CHECK(
        cudaFree(d_v_in)
    );

    CUDA_CHECK(
        cudaFree(d_v_out)
    );

    CUDA_CHECK(
        cudaFree(d_cos)
    );

    CUDA_CHECK(
        cudaFree(d_sin)
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
        << "Q heads            : "
        << HEADS << "\n";

    std::cout
        << "KV heads           : "
        << KV_HEADS << "\n";

    std::cout
        << "Head dimension     : "
        << HEAD_DIM << "\n";

    std::cout
        << "Rotary dimension   : "
        << ROTARY_DIM << "\n";

    std::cout
        << "Operations         : Q RoPE + K RoPE + V Transpose\n";

    std::cout
        << "Position offset    : 0\n";

    std::cout
        << "Warmup iterations  : "
        << WARMUP_ITERS << "\n";

    std::cout
        << "Benchmark iters    : "
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
