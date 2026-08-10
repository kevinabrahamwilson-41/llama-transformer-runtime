#include "tokenizer.hpp"
#include "model_loader.hpp"
#include "special_tokens.hpp"
#include "regex_split.hpp"
#include "bpe.hpp"
#include <stdexcept>
namespace transformer::tokenizer{
    Tokenizer::Tokenizer(
        const std::string& model_path
    ):
    model_(
        ModelLoader::load(model_path)
    ){
        SpecialTokenLoader loader;
        loader.load(model_);
        initialize_special_token_ids();
    }
    void Tokenizer::initialize_special_token_ids(){
        auto get_id =
            [this](const std::string& token)->TokenID{
                auto it =
                    model_.special_tokens.find(token);
                if(it == model_.special_tokens.end())
                {
                    throw std::runtime_error(
                        "Missing required special token: " + token
                    );
                }
                return it->second;
            };
        bos_id_ =
            get_id("<|begin_of_text|>");
        eos_id_ =
            get_id("<|end_of_text|>");
        pad_id_ =
            get_id("<|finetune_right_pad_id|>");
        eot_id_ =
            get_id("<|eot_id|>");
        eom_id_ =
            get_id("<|eom_id|>");
        python_tag_id_ =
            get_id("<|python_tag|>");
            start_header_id_ =
            get_id("<|start_header_id|>");
        end_header_id_ =
            get_id("<|end_header_id|>");
        stop_tokens_.clear();
        stop_tokens_.push_back(
            eos_id_
        );
        stop_tokens_.push_back(
            eom_id_
        );
        stop_tokens_.push_back(
            eot_id_
        );
    }
    TokenSequence Tokenizer::encode(
        const std::string& text,
        const EncodeOptions& options
    ) const{
        TokenSequence output;
        if(options.bos){
            output.push_back(bos_id_);
        }
        size_t pos = 0;
        while(pos < text.size()){
            bool found_special = false;
            // Check if any special token starts here
            for(const auto& [token, id] : model_.special_tokens){
                if(text.compare(pos, token.size(), token) == 0){
                    output.push_back(id);
                    pos += token.size();
                    found_special = true;
                    break;
                }
            }
            if(found_special)
                continue;
            // Find next special token boundary
            size_t next_special = text.size();
            for(const auto& [token, id] : model_.special_tokens){
                size_t found = text.find(token, pos);
                if(found != std::string::npos &&
                found < next_special){
                    next_special = found;
                }
            }
            // Normal text before special token
            std::string normal_piece =
                text.substr(pos, next_special - pos);
            if(!normal_piece.empty()){
                auto pieces =
                    RegexSplitter::split(normal_piece);
                for(const auto& piece : pieces){
                    auto tokens =
                        BPE::encode_piece(
                            piece,
                            model_
                        );
                    output.insert(
                        output.end(),
                        tokens.begin(),
                        tokens.end()
                    );
                }
            }
            pos = next_special;
        }
        if(options.eos){
            output.push_back(eos_id_);
        }
        return output;
    }
    std::string Tokenizer::decode(
        const TokenSequence& tokens
    ) const{
        std::string output;
        for(TokenID id : tokens){
            if(id < 0 ||
            static_cast<size_t>(id) >= model_.vocabulary.size())
            {
                throw std::runtime_error(
                    "Invalid token id during decode"
                );
            }
            output += model_.vocabulary[id];
        }
        return output;
    }
    TokenID Tokenizer::bos_id() const noexcept{
        return bos_id_;
    }
    TokenID Tokenizer::eos_id() const noexcept{
        return eos_id_;
    }
    TokenID Tokenizer::pad_id() const noexcept{
        return pad_id_;
    }
    TokenID Tokenizer::eot_id() const noexcept{
        return eot_id_;
    }
    TokenID Tokenizer::eom_id() const noexcept{
        return eom_id_;
    }
    TokenID Tokenizer::python_tag_id() const noexcept{
        return python_tag_id_;
    }
    TokenID Tokenizer::start_header_id() const noexcept{
    return start_header_id_;
    }
    TokenID Tokenizer::end_header_id() const noexcept{
        return end_header_id_;
    }
    const TokenizerModel& Tokenizer::model() const noexcept{
        return model_;
    }
    std::size_t Tokenizer::vocab_size() const noexcept{
        return model_.vocab_size();
    }
}