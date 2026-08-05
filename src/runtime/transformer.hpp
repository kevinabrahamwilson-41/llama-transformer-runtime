#pragma once
#include "transformer_block.hpp"
#include "tensor.hpp"
#include "weights.hpp"
#include <vector>
namespace runtime{
    class Transformer{
        private:
            static constexpr int NUM_LAYERS = 16;
            static constexpr int HIDDEN_SIZE = 2048;
            std::vector<TransformerBlock> layers_;
        public:
            Transformer(
                llama::LlamaWeights& weights,
                float* cos_table,
                float* sin_table,
                int max_seq_len
            );
            void forward(
                const Tensor& input,
                Tensor& output,
                int position
            );
    };
}