#pragma once
#include "token.hpp"
#include <string>
#include <string_view>
#include <cstddef>
#include <vector>
namespace transformer::tokenizer{
    class Tokenizer{
    public:
        explicit Tokenizer(const std::string& model_path);
        TokenSequence encode(
            const std::string& text,
            const EncodeOptions& options={}
        ) const;
        std::string decode(
            const TokenSequence& tokens
        ) const;
        [[nodiscard]]
        TokenID bos_id() const noexcept;
        [[nodiscard]]
        TokenID eos_id() const noexcept;
        [[nodiscard]]
        TokenID pad_id() const noexcept;
        [[nodiscard]]
        TokenID eot_id() const noexcept;
        [[nodiscard]]
        TokenID eom_id() const noexcept;
        [[nodiscard]]
        TokenID python_tag_id() const noexcept;
        [[nodiscard]]
        const TokenizerModel& model() const noexcept;
        [[nodiscard]]
        std::size_t vocab_size() const noexcept;
    private:
        void initialize_special_token_ids();
    private:
        TokenizerModel model_;
        TokenID bos_id_ = -1;
        TokenID eos_id_ = -1;
        TokenID pad_id_ = -1;
        TokenID eot_id_ = -1;
        TokenID eom_id_ = -1;
        TokenID python_tag_id_ = -1;
        TokenSequence stop_tokens_;
    };
}