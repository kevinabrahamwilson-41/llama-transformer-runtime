#pragma once
#include <cstdint>
#include <string>
#include <string_view>
namespace transformer::tokenizer{
    class UTF8{
    public:
        static bool is_space(uint32_t cp);
        static bool is_letter(uint32_t cp);
        static bool is_digit(uint32_t cp);
        static uint32_t decode(
            std::string_view text,
            size_t& index
        );
    };
    }