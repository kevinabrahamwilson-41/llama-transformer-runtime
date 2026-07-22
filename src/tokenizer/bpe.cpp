#include "bpe.hpp"
#include "token.hpp"
#include "utf8.hpp"
#include <stdexcept>
namespace transformer::tokenizer{
    std::vector<BytePiece> BPE::split_into_bytes(std::string_view piece){
        std::vector<BytePiece> bytes;
        bytes.reserve(piece.size());
        for(unsigned char c : piece){
            bytes.emplace_back(1, static_cast<char>(c));
        }
        return bytes;
    }
    MergeCandidate BPE::find_best_merge(
        const std::vector<BytePiece>& pieces,
        const MergeableRanks& mergeable_ranks
    ){
        MergeCandidate best{
            0,
            0,
            std::numeric_limits<Rank>::max()
        };

        if(pieces.size()<2)
            return best;

        for(std::size_t i=0;i+1<pieces.size();i++){

            BytePiece pair;
            pair.reserve(
                pieces[i].size()+
                pieces[i+1].size()
            );

            pair+=pieces[i];
            pair+=pieces[i+1];

            auto it=mergeable_ranks.find(pair);

            if(it==mergeable_ranks.end())
                continue;

            if(it->second<best.rank){
                best.left_index=i;
                best.right_index=i+1;
                best.rank=it->second;
            }
        }
        return best;
    }
    void BPE::apply_merge(
        std::vector<BytePiece>& pieces,
        const MergeCandidate& candidate
    ){
        pieces[candidate.left_index] +=
            pieces[candidate.right_index];
        pieces.erase(
            pieces.begin() +
            candidate.right_index
        );
    }
    TokenSequence BPE::lookup_ids(
        const std::vector<BytePiece>& pieces,
        const TokenizerModel& model
    ){
        TokenSequence ids;
        ids.reserve(pieces.size());
        for(const auto& piece : pieces){
            auto it=model.mergeable_ranks.find(piece);
            if(it==model.mergeable_ranks.end()){
                throw std::runtime_error(
                    "Unknown byte piece encountered during BPE lookup"
                );
            }
            ids.push_back(
                static_cast<TokenID>(it->second)
            );
        }
        return ids;
    }
    bool BPE::has_merge(
        const MergeCandidate& candidate
    ){
        return candidate.rank !=
            std::numeric_limits<Rank>::max();
    }
    TokenSequence BPE::encode_piece(
        std::string_view piece,
        const TokenizerModel& model
    ){
        if(piece.empty())
            return {};
        std::vector<BytePiece> pieces=
            split_into_bytes(piece);
        while(pieces.size()>1){
            MergeCandidate candidate=
                find_best_merge(
                    pieces,
                    model.mergeable_ranks
                );
            if(!has_merge(candidate))
                break;
            apply_merge(
                pieces,
                candidate
            );
        }
        return lookup_ids(
            pieces,
            model
        );
    }
}
