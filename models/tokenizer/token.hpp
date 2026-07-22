#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <limits>
#include <vector>
#include <unordered_map>
#include <unordered_set>
namespace transformer::tokenizer{
    using TokenID=std::int32_t;
    using Rank=std::int32_t;
    using BytePiece=std::string;
    using MergeableRanks=std::unordered_map<BytePiece,Rank>;
    using Vocabulary = std::vector<BytePiece>;
    using TokenToID = std::unordered_map<std::string,TokenID>;
    struct TokenizerModel{
        MergeableRanks mergeable_ranks;
        Vocabulary vocabulary;
        TokenToID special_tokens;
        Vocabulary special_vocabulary;
        [[nodiscard]]
        std::size_t vocab_size() const noexcept{
            return vocabulary.size()+special_vocabulary.size();
        }
    };
    struct MergeCandidate{
        std::size_t left_index=0;
        std::size_t right_index=0;
        Rank rank=std::numeric_limits<Rank>::max();
        bool valid=false;
    };
    struct EncodeOptions{
        bool bos=true;
        bool eos=false;
    };
    enum class SpecialTokenPolicy{
        None,
        AllowAll,
        ErrorOnSpecial
    };
    using SplitPieces=std::vector<std::string>;
    using TokenSequence=std::vector<TokenID>;
}