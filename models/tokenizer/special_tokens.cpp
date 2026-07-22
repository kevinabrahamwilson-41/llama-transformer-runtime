#include "special_tokens.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
namespace transformer::tokenizer {
    void SpecialTokenLoader::load(
        TokenizerModel& model){
        std::ifstream file(
            "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama-3.2-1b-instruct/tokenizer.json"
        );
        if(!file.is_open()){
            throw std::runtime_error(
                "Cannot open tokenizer.json"
            );
        }
        nlohmann::json tokenizer_json;
        file >> tokenizer_json;
        size_t count = 0;
        for(const auto& token : tokenizer_json["added_tokens"]){
            if(!token["special"].get<bool>())
                continue;
            TokenID id =
                token["id"].get<TokenID>();
            std::string content =
                token["content"].get<std::string>();
            auto [it, inserted] =
                model.special_tokens.emplace(
                    content,
                    id
                );
            if(!inserted){
                throw std::runtime_error(
                    "Duplicate special token"
                );
            }
            if(model.vocabulary.size() <= static_cast<size_t>(id)){
                model.vocabulary.resize(id + 1);
            }
            model.vocabulary[id] = content;
            count++;
        }
        std::cout
            << "[Special Token Loader] Added "
            << count
            << " special tokens\n";
        std::cout
            << "[Tokenizer] Final vocabulary size: "
            << model.vocab_size()
            << "\n";
    }
}