#include "../../src/runtime/weights.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr const char* WEIGHTS_PATH =
    "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/"
    "weights/llama_weights.bin";

constexpr const char* REFERENCE_PATH =
    "/tmp/weights_reference.json";

constexpr int NUM_LAYERS = 16;
constexpr int HIDDEN_SIZE = 2048;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 64;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int VOCAB_SIZE = 128256;

constexpr std::size_t BF16_BYTES =
    sizeof(__nv_bfloat16);

struct TensorView
{
    const char* name;
    const __nv_bfloat16* data;
    std::size_t elements;
};

void check_cuda(
    cudaError_t error,
    const char* operation
)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) +
            " failed: " +
            cudaGetErrorString(error)
        );
    }
}

std::string read_file(
    const char* path
)
{
    std::ifstream file(path);

    if (!file)
    {
        throw std::runtime_error(
            std::string("Failed to open reference file: ") +
            path
        );
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

std::string sha256_raw(
    const std::vector<std::uint8_t>& data
)
{
    /*
     * We intentionally do not implement SHA256 here.
     *
     * The CUDA test instead dumps the raw loaded bytes.
     * The Python reference remains the authoritative raw-bit
     * manifest.
     */

    return {};
}

void check_tensor(
    const TensorView& tensor,
    int index
)
{
    std::vector<__nv_bfloat16> host(
        tensor.elements
    );

    check_cuda(
        cudaMemcpy(
            host.data(),
            tensor.data,
            tensor.elements * BF16_BYTES,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy tensor"
    );

    std::printf(
        "[%03d] %-70s elements=%zu\n",
        index,
        tensor.name,
        tensor.elements
    );

    std::printf(
        "      first:"
    );

    const std::size_t first_count =
        tensor.elements < 8
            ? tensor.elements
            : 8;

    for (std::size_t i = 0;
         i < first_count;
         ++i)
    {
        const std::uint16_t bits =
            *reinterpret_cast<
                const std::uint16_t*
            >(&host[i]);

        std::printf(
            " %04x",
            bits
        );
    }

    std::printf("\n");

    std::printf(
        "      last :"
    );

    const std::size_t start =
        tensor.elements > 8
            ? tensor.elements - 8
            : 0;

    for (std::size_t i = start;
         i < tensor.elements;
         ++i)
    {
        const std::uint16_t bits =
            *reinterpret_cast<
                const std::uint16_t*
            >(&host[i]);

        std::printf(
            " %04x",
            bits
        );
    }

    std::printf("\n");
}

} // namespace


int main()
{
    try
    {
        std::printf(
            "============================================================\n"
        );

        std::printf(
            "CUDA WEIGHTS LOADER TEST\n"
        );

        std::printf(
            "============================================================\n\n"
        );

        llama::LlamaWeights weights{};

        llama::load_weights(
            WEIGHTS_PATH,
            weights
        );

        std::printf(
            "\n[CUDA] Weights loaded.\n\n"
        );

        int index = 0;

        // ====================================================
        // EMBEDDING
        // ====================================================

        check_tensor(
            {
                "model.embed_tokens.weight",
                weights.embed_tokens,
                static_cast<std::size_t>(
                    VOCAB_SIZE
                ) * HIDDEN_SIZE
            },
            index++
        );

        // ====================================================
        // LAYERS
        // ====================================================

        for (int layer = 0;
             layer < NUM_LAYERS;
             ++layer)
        {
            const auto& w =
                weights.layers[layer];

            std::string input_name =
                "model.layers." +
                std::to_string(layer) +
                ".input_layernorm.weight";

            check_tensor(
                {
                    input_name.c_str(),
                    w.input_layernorm,
                    HIDDEN_SIZE
                },
                index++
            );

            std::string down_name =
                "model.layers." +
                std::to_string(layer) +
                ".mlp.down_proj.weight";

            check_tensor(
                {
                    down_name.c_str(),
                    w.down_proj,
                    static_cast<std::size_t>(
                        HIDDEN_SIZE
                    ) * INTERMEDIATE_SIZE
                },
                index++
            );

            std::string gate_name =
                "model.layers." +
                std::to_string(layer) +
                ".mlp.gate_proj.weight";

            check_tensor(
                {
                    gate_name.c_str(),
                    w.gate_proj,
                    static_cast<std::size_t>(
                        INTERMEDIATE_SIZE
                    ) * HIDDEN_SIZE
                },
                index++
            );

            std::string up_name =
                "model.layers." +
                std::to_string(layer) +
                ".mlp.up_proj.weight";

            check_tensor(
                {
                    up_name.c_str(),
                    w.up_proj,
                    static_cast<std::size_t>(
                        INTERMEDIATE_SIZE
                    ) * HIDDEN_SIZE
                },
                index++
            );

            std::string post_norm_name =
                "model.layers." +
                std::to_string(layer) +
                ".post_attention_layernorm.weight";

            check_tensor(
                {
                    post_norm_name.c_str(),
                    w.post_attention_layernorm,
                    HIDDEN_SIZE
                },
                index++
            );

            std::string k_name =
                "model.layers." +
                std::to_string(layer) +
                ".self_attn.k_proj.weight";

            check_tensor(
                {
                    k_name.c_str(),
                    w.k_proj,
                    static_cast<std::size_t>(
                        HIDDEN_SIZE
                    ) *
                    NUM_KV_HEADS *
                    HEAD_DIM
                },
                index++
            );

            std::string o_name =
                "model.layers." +
                std::to_string(layer) +
                ".self_attn.o_proj.weight";

            check_tensor(
                {
                    o_name.c_str(),
                    w.o_proj,
                    static_cast<std::size_t>(
                        HIDDEN_SIZE
                    ) * HIDDEN_SIZE
                },
                index++
            );

            std::string q_name =
                "model.layers." +
                std::to_string(layer) +
                ".self_attn.q_proj.weight";

            check_tensor(
                {
                    q_name.c_str(),
                    w.q_proj,
                    static_cast<std::size_t>(
                        HIDDEN_SIZE
                    ) * HIDDEN_SIZE
                },
                index++
            );

            std::string v_name =
                "model.layers." +
                std::to_string(layer) +
                ".self_attn.v_proj.weight";

            check_tensor(
                {
                    v_name.c_str(),
                    w.v_proj,
                    static_cast<std::size_t>(
                        HIDDEN_SIZE
                    ) *
                    NUM_KV_HEADS *
                    HEAD_DIM
                },
                index++
            );
        }

        // ====================================================
        // FINAL NORM
        // ====================================================

        check_tensor(
            {
                "model.norm.weight",
                weights.final_norm,
                HIDDEN_SIZE
            },
            index++
        );

        std::printf(
            "\n[CUDA] Tensor count: %d\n",
            index
        );

        llama::free_weights(
            weights
        );

        std::printf(
            "\n============================================================\n"
        );

        std::printf(
            "CUDA WEIGHTS TEST COMPLETE\n"
        );

        std::printf(
            "============================================================\n"
        );
    }
    catch (const std::exception& error)
    {
        std::fprintf(
            stderr,
            "\nFATAL: %s\n",
            error.what()
        );

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}