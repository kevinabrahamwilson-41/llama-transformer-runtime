#pragma once
#include "token.hpp"
#include <string>
namespace transformer::tokenizer{
    class ModelLoader{
    public:
        static TokenizerModel load(
            const std::string& path
        );
    };
}