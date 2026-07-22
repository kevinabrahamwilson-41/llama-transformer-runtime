#pragma once

#include "token.hpp"

namespace transformer::tokenizer {

class SpecialTokenLoader {
public:
    static void load(TokenizerModel& model);
};

}