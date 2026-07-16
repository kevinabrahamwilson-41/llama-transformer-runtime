#include "weights.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama
{

// ============================================================
// CUDA ERROR CHECKING
// ============================================================

#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t error = (call);                           \
                                                              \
        if (error != cudaSuccess)                             \
        {                                                     \
            throw std::runtime_error(                         \
                std::string("CUDA error at ") +               \
                __FILE__ + ":" +                              \
                std::to_string(__LINE__) +                    \
                " -> " +                                      \
                cudaGetErrorString(error)                     \
            );                                                \
        }                                                     \
    } while (0)


// ============================================================
// MODEL WEIGHT SIZES
// ============================================================

constexpr std::size_t BF16_BYTES =
    sizeof(__nv_bfloat16);

constexpr std::size_t EMBED_ELEMENTS =
    static_cast<std::size_t>(VOCAB_SIZE) *
    HIDDEN_SIZE;

constexpr std::size_t NORM_ELEMENTS =
    HIDDEN_SIZE;

constexpr std::size_t Q_PROJ_ELEMENTS =
    static_cast<std::size_t>(HIDDEN_SIZE) *
    HIDDEN_SIZE;

constexpr std::size_t K_PROJ_ELEMENTS =
    static_cast<std::size_t>(NUM_KV_HEADS) *
    HEAD_DIM *
    HIDDEN_SIZE;

constexpr std::size_t V_PROJ_ELEMENTS =
    static_cast<std::size_t>(NUM_KV_HEADS) *
    HEAD_DIM *
    HIDDEN_SIZE;

constexpr std::size_t O_PROJ_ELEMENTS =
    static_cast<std::size_t>(HIDDEN_SIZE) *
    HIDDEN_SIZE;

constexpr std::size_t GATE_PROJ_ELEMENTS =
    static_cast<std::size_t>(INTERMEDIATE_SIZE) *
    HIDDEN_SIZE;

constexpr std::size_t UP_PROJ_ELEMENTS =
    static_cast<std::size_t>(INTERMEDIATE_SIZE) *
    HIDDEN_SIZE;

constexpr std::size_t DOWN_PROJ_ELEMENTS =
    static_cast<std::size_t>(HIDDEN_SIZE) *
    INTERMEDIATE_SIZE;


// ============================================================
// EXPECTED BINARY SIZE
// ============================================================

constexpr std::size_t LAYER_BYTES =
    (
        NORM_ELEMENTS +

        DOWN_PROJ_ELEMENTS +
        GATE_PROJ_ELEMENTS +
        UP_PROJ_ELEMENTS +

        NORM_ELEMENTS +

        K_PROJ_ELEMENTS +
        O_PROJ_ELEMENTS +
        Q_PROJ_ELEMENTS +
        V_PROJ_ELEMENTS
    ) * BF16_BYTES;


constexpr std::size_t EXPECTED_WEIGHTS_BYTES =
    EMBED_ELEMENTS * BF16_BYTES +

    NUM_LAYERS * LAYER_BYTES +

    NORM_ELEMENTS * BF16_BYTES;


// ============================================================
// WEIGHT ALLOCATION
// ============================================================

static void allocate_layer(
    TransformerLayerWeights& layer
)
{
    CUDA_CHECK(cudaMalloc(
        &layer.input_layernorm,
        NORM_ELEMENTS * BF16_BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &layer.q_proj,
        Q_PROJ_ELEMENTS * BF16_BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &layer.k_proj,
        K_PROJ_ELEMENTS * BF16_BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &layer.v_proj,
        V_PROJ_ELEMENTS * BF16_BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &layer.o_proj,
        O_PROJ_ELEMENTS * BF16_BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &layer.post_attention_layernorm,
        NORM_ELEMENTS * BF16_BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &layer.gate_proj,
        GATE_PROJ_ELEMENTS * BF16_BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &layer.up_proj,
        UP_PROJ_ELEMENTS * BF16_BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &layer.down_proj,
        DOWN_PROJ_ELEMENTS * BF16_BYTES
    ));
}


// ============================================================
// FREE ONE LAYER
// ============================================================

static void free_layer(
    TransformerLayerWeights& layer
)
{
    cudaFree(layer.input_layernorm);
    cudaFree(layer.q_proj);
    cudaFree(layer.k_proj);
    cudaFree(layer.v_proj);
    cudaFree(layer.o_proj);

    cudaFree(layer.post_attention_layernorm);

    cudaFree(layer.gate_proj);
    cudaFree(layer.up_proj);
    cudaFree(layer.down_proj);

    layer.input_layernorm = nullptr;

    layer.q_proj = nullptr;
    layer.k_proj = nullptr;
    layer.v_proj = nullptr;
    layer.o_proj = nullptr;

    layer.post_attention_layernorm = nullptr;

    layer.gate_proj = nullptr;
    layer.up_proj = nullptr;
    layer.down_proj = nullptr;
}


// ============================================================
// LOAD ONE TENSOR
// ============================================================

static void load_tensor(
    std::ifstream& file,
    __nv_bfloat16* destination,
    std::size_t elements,
    std::vector<std::uint8_t>& host_buffer
)
{
    const std::size_t bytes =
        elements * BF16_BYTES;


    file.read(
        reinterpret_cast<char*>(host_buffer.data()),
        static_cast<std::streamsize>(bytes)
    );


    if (!file)
    {
        throw std::runtime_error(
            "Failed to read tensor from weights file"
        );
    }


    CUDA_CHECK(cudaMemcpy(
        destination,
        host_buffer.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));
}


// ============================================================
// LOAD WEIGHTS
// ============================================================

void load_weights(
    const char* weights_path,
    LlamaWeights& weights
)
{
    std::printf(
        "[WEIGHTS] Loading Llama 3.2 1B weights...\n"
    );


    std::ifstream file(
        weights_path,
        std::ios::binary
    );


    if (!file)
    {
        throw std::runtime_error(
            std::string(
                "Failed to open weights file: "
            ) + weights_path
        );
    }


    // --------------------------------------------------------
    // Validate file size
    // --------------------------------------------------------

    file.seekg(
        0,
        std::ios::end
    );


    const std::uint64_t file_size =
        static_cast<std::uint64_t>(
            file.tellg()
        );


    file.seekg(
        0,
        std::ios::beg
    );


    if (file_size != EXPECTED_WEIGHTS_BYTES)
    {
        throw std::runtime_error(
            "Invalid weights file size. Expected " +
            std::to_string(EXPECTED_WEIGHTS_BYTES) +
            " bytes, got " +
            std::to_string(file_size) +
            " bytes"
        );
    }


    std::printf(
        "[WEIGHTS] File size validated: %llu bytes\n",
        static_cast<unsigned long long>(
            file_size
        )
    );


    // --------------------------------------------------------
    // Host staging buffer
    //
    // Largest tensor:
    // 8192 x 2048 BF16
    // --------------------------------------------------------

    std::vector<std::uint8_t> host_buffer(
        DOWN_PROJ_ELEMENTS * BF16_BYTES
    );


    // --------------------------------------------------------
    // Allocate embedding
    // --------------------------------------------------------

    CUDA_CHECK(cudaMalloc(
        &weights.embed_tokens,
        EMBED_ELEMENTS * BF16_BYTES
    ));


    // --------------------------------------------------------
    // Allocate layers
    // --------------------------------------------------------

    for (int layer = 0;
         layer < NUM_LAYERS;
         ++layer)
    {
        allocate_layer(
            weights.layers[layer]
        );
    }


    // --------------------------------------------------------
    // Allocate final norm
    // --------------------------------------------------------

    CUDA_CHECK(cudaMalloc(
        &weights.final_norm,
        NORM_ELEMENTS * BF16_BYTES
    ));


    // --------------------------------------------------------
    // EMBEDDING
    // --------------------------------------------------------

    std::printf(
        "[WEIGHTS] Loading embedding...\n"
    );


    load_tensor(
        file,
        weights.embed_tokens,
        EMBED_ELEMENTS,
        host_buffer
    );


    // --------------------------------------------------------
    // TRANSFORMER LAYERS
    // --------------------------------------------------------

    for (int layer = 0;
         layer < NUM_LAYERS;
         ++layer)
    {
        TransformerLayerWeights& w =
            weights.layers[layer];


        std::printf(
            "[WEIGHTS] Loading layer %d/%d\n",
            layer + 1,
            NUM_LAYERS
        );


        // ----------------------------------------------------
        // EXACT CONVERTER ORDER
        // ----------------------------------------------------

        load_tensor(
            file,
            w.input_layernorm,
            NORM_ELEMENTS,
            host_buffer
        );


        load_tensor(
            file,
            w.down_proj,
            DOWN_PROJ_ELEMENTS,
            host_buffer
        );


        load_tensor(
            file,
            w.gate_proj,
            GATE_PROJ_ELEMENTS,
            host_buffer
        );


        load_tensor(
            file,
            w.up_proj,
            UP_PROJ_ELEMENTS,
            host_buffer
        );


        load_tensor(
            file,
            w.post_attention_layernorm,
            NORM_ELEMENTS,
            host_buffer
        );


        load_tensor(
            file,
            w.k_proj,
            K_PROJ_ELEMENTS,
            host_buffer
        );


        load_tensor(
            file,
            w.o_proj,
            O_PROJ_ELEMENTS,
            host_buffer
        );


        load_tensor(
            file,
            w.q_proj,
            Q_PROJ_ELEMENTS,
            host_buffer
        );


        load_tensor(
            file,
            w.v_proj,
            V_PROJ_ELEMENTS,
            host_buffer
        );
    }


    // --------------------------------------------------------
    // FINAL RMSNORM
    // --------------------------------------------------------

    std::printf(
        "[WEIGHTS] Loading final norm...\n"
    );


    load_tensor(
        file,
        weights.final_norm,
        NORM_ELEMENTS,
        host_buffer
    );


    // --------------------------------------------------------
    // TIED EMBEDDINGS
    // --------------------------------------------------------

    weights.lm_head =
        weights.embed_tokens;


    std::printf(
        "[WEIGHTS] All weights loaded successfully.\n"
    );
}


// ============================================================
// FREE ALL WEIGHTS
// ============================================================

void free_weights(
    LlamaWeights& weights
)
{
    if (weights.embed_tokens != nullptr)
    {
        CUDA_CHECK(cudaFree(
            weights.embed_tokens
        ));

        weights.embed_tokens = nullptr;
    }


    for (int layer = 0;
         layer < NUM_LAYERS;
         ++layer)
    {
        free_layer(
            weights.layers[layer]
        );
    }


    if (weights.final_norm != nullptr)
    {
        CUDA_CHECK(cudaFree(
            weights.final_norm
        ));

        weights.final_norm = nullptr;
    }


    // lm_head aliases embed_tokens.
    // Do NOT cudaFree it separately.
    weights.lm_head = nullptr;


    std::printf(
        "[WEIGHTS] GPU weights freed.\n"
    );
}

} // namespace llama