#pragma once
#include "token.hpp"
#include <string>
namespace transformer::tokenizer{
class RegexSplitter{
public:
    static SplitPieces split(
        const std::string& text
    );
};
}