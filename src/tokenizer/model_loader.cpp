#include "model_loader.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <iomanip>
namespace transformer::tokenizer {
    namespace{
    /*
        Base64 decoder
        tokenizer.model stores token bytes as Base64 strings.
        Example:
            IQ== 0
        becomes:
            "!" -> rank 0
    */
    static constexpr char BASE64_TABLE[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    std::string base64_decode(const std::string& input){
        static std::array<int, 256> reverse_table{};
        static bool initialized = false;
        if (!initialized){
            reverse_table.fill(-1);
            for (int i = 0; i < 64; i++){
                reverse_table[
                    static_cast<unsigned char>(BASE64_TABLE[i])
                ] = i;
            }
            initialized = true;
        }
        std::string output;
        output.reserve(
            (input.size() * 3) / 4
        );
        int buffer = 0;
        int bits = 0;
        for (unsigned char c : input){
            if (c == '=')
                break;
            const int value =
                reverse_table[c];
            if (value == -1){
                throw std::runtime_error(
                    "Invalid Base64 character in tokenizer.model"
                );
            }
            buffer = (buffer << 6) | value;
            bits += 6;
            if (bits >= 8){
                bits -= 8;
                output.push_back(
                    static_cast<char>(
                        (buffer >> bits) & 0xFF
                    )
                );
            }
        }
        return output;
    }
    }
    /*
        Loads Llama tiktoken BPE vocabulary.
        File format:
            base64_token     rank
        Example:
            IQ==             0
            Ig==             1
            Iw==             2

    */
    TokenizerModel ModelLoader::load(
        const std::string& path
    ){
        std::ifstream file(
            path,
            std::ios::in
        );
        if (!file.is_open())
        {
            throw std::runtime_error(
                "Cannot open tokenizer model: " + path
            );
        }
        TokenizerModel model;
        std::string encoded_token;
        Rank rank;
        while (file >> encoded_token >> rank)
        {
            /*
                Convert:
                    IQ==
                into:
                    !
            */
            const std::string token_bytes =
                base64_decode(encoded_token);
            /*
                bytes -> rank
                Used during encoding:
                    text
                    |
                    v
                BPE merge
                    |
                    v
                token bytes
                    |
                    v
                rank lookup
            */
            auto [it1, inserted1] =
                model.mergeable_ranks.emplace(
                    token_bytes,
                    rank
                );
            if (!inserted1){
                throw std::runtime_error(
                    "Duplicate token found in tokenizer.model"
                );
            }
            /*
                rank -> bytes
                Used during decoding:
                    token id
                        |
                        v
                token bytes
                        |
                        v
                    text
            */
            if(model.vocabulary.size() <= static_cast<size_t>(rank)){
                model.vocabulary.resize(rank + 1);
            }
            model.vocabulary[rank] = token_bytes;
        }
        if (model.mergeable_ranks.empty())
        {
            throw std::runtime_error(
                "Tokenizer model loaded zero tokens"
            );
        }
        std::cout
            << "[Tokenizer Loader] Loaded vocabulary: "
            << model.vocab_size()
            << " tokens\n";
        return model;
    }
} // namespace transformer::tokenizer
