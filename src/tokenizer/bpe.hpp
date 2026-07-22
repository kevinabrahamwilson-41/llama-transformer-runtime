#pragma once
#include "token.hpp"
#include <string_view>
#include <vector>
namespace transformer::tokenizer{
    class BPE{
    public:
        static TokenSequence encode_piece(
            std::string_view piece,
            const TokenizerModel& model
        );
    private:
        static std::vector<BytePiece> split_into_bytes(
            std::string_view piece
        );
        static MergeCandidate find_best_merge(
            const std::vector<BytePiece>& pieces,
            const MergeableRanks& mergeable_ranks
        );
        static bool has_merge(
            const MergeCandidate& candidate
        );
        static void apply_merge(
            std::vector<BytePiece>& pieces,
            const MergeCandidate& candidate
        );
        static TokenSequence lookup_ids(
            const std::vector<BytePiece>& pieces,
            const TokenizerModel& model
        );
    };
}