#include "regex_split.hpp"
#include "utf8.hpp"
#include <string>
#include <iostream>
namespace transformer::tokenizer {
static inline bool is_contraction_start(uint32_t cp) {
    return cp == 's' || cp == 't' || cp == 'm' || cp == 'd';
}
static inline uint32_t lower_ascii(uint32_t cp){
    if(cp >= 'A' && cp <= 'Z')
        return cp + 32;
    return cp;
}
static inline bool is_contraction_pair(uint32_t a, uint32_t b) {
    return (a == 'r' && b == 'e') ||
           (a == 'v' && b == 'e') ||
           (a == 'l' && b == 'l');
}
SplitPieces RegexSplitter::split(const std::string& text) {
    SplitPieces pieces;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        size_t start = i;
        uint32_t cp = UTF8::decode(text, i); // i is advanced to the next char position
        // 1) contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp == '\'') {
            size_t tmp = i;
            if (tmp < n) {
                uint32_t next = lower_ascii(UTF8::decode(text, tmp));
                if (is_contraction_start(next)) {
                    i = tmp;
                    pieces.emplace_back(text.substr(start, i - start));
                    continue;
                }
                if ((next == 'r' || next == 'v' || next == 'l') && tmp < n) {
                    size_t tmp2 = tmp;
                    uint32_t second = lower_ascii(UTF8::decode(text, tmp2));
                    if (is_contraction_pair(next, second)) {
                        i = tmp2;
                        pieces.emplace_back(text.substr(start, i - start));
                        continue;
                    }
                }
            }
        }
        // Reset to state before current character read for pattern matching standard structures
        i = start; 
        size_t cur = start;
        cp = UTF8::decode(text, cur);
        // 2) [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t tmp = start;
            uint32_t first_cp = UTF8::decode(text, tmp);
            bool has_prefix = false;

            if (!UTF8::is_letter(first_cp) && !UTF8::is_digit(first_cp) && first_cp != '\r' && first_cp != '\n') {
                has_prefix = true;
            }
            size_t letter_pos = has_prefix ? tmp : start;
            if (letter_pos < n) {
                size_t test_pos = letter_pos;
                uint32_t next = UTF8::decode(text, test_pos);
                if (UTF8::is_letter(next)) {
                    // Match sequence of letters
                    letter_pos = test_pos;
                    while (letter_pos < n) {
                        size_t old = letter_pos;
                        uint32_t c = UTF8::decode(text, letter_pos);
                        if (!UTF8::is_letter(c)) {
                            letter_pos = old;
                            break;
                        }
                    }
                    i = letter_pos;
                    pieces.emplace_back(text.substr(start, i - start));
                    continue;
                }
            }
        }
        // 3) \p{N}{1,3}
        if (UTF8::is_digit(cp)) {
            i = cur;
            int count = 1;
            while (i < n && count < 3) {
                size_t old = i;
                uint32_t c = UTF8::decode(text, i);
                if (!UTF8::is_digit(c)) {
                    i = old;
                    break;
                }
                count++;
            }
            pieces.emplace_back(text.substr(start, i - start));
            continue;
        }
        // 4) newline handling
        if (cp == '\r' || cp == '\n') {
            i = cur;
            while (i < n) {
                size_t old = i;
                uint32_t c = UTF8::decode(text, i);
                if (c != '\r' && c != '\n') {
                    i = old;
                    break;
                }
            }
            pieces.emplace_back(text.substr(start, i - start));
            continue;
        }
        // 5) ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t punct_start = (cp == ' ') ? cur : start;
            i = punct_start;
            // consume punctuation run
            while (i < n)
            {
                size_t old = i;
                uint32_t c = UTF8::decode(text, i);

                if (UTF8::is_letter(c) ||
                    UTF8::is_digit(c) ||
                    UTF8::is_space(c))
                {
                    i = old;
                    break;
                }
            }
            if (i > punct_start){
                // consume trailing newline only
                while (i < n){
                    size_t old = i;
                    uint32_t c = UTF8::decode(text, i);
                    if (c == '\r' || c == '\n')
                        continue;
                    i = old;
                    break;
                }
                pieces.emplace_back(
                    text.substr(start, i-start)
                );
                continue;
            }
        }
        // 6) punctuation: [^\s\p{L}\p{N}]+
        if (!UTF8::is_letter(cp) && !UTF8::is_digit(cp) && !UTF8::is_space(cp) && cp != '\r' && cp != '\n') {
            i = cur;
            while (i < n) {
                size_t old = i;
                uint32_t c = UTF8::decode(text, i);
                if (UTF8::is_letter(c) || UTF8::is_digit(c) || UTF8::is_space(c) || c == '\r' || c == '\n') {
                    i = old;
                    break;
                }
            }
            pieces.emplace_back(text.substr(start, i - start));
            continue;
        }
                // \s+(?!\S) or \s+
        if (UTF8::is_space(cp)){
            i = cur;
            size_t last_space_start = start; // offset of the most recently consumed space char
            while (i < n){
                size_t old = i;
                uint32_t c = UTF8::decode(text, i);
                if (!UTF8::is_space(c)){
                    i = old;
                    break;
                }
                last_space_start = old;
            }
            // i now points just past the full whitespace run [start, i).
            // \s+(?!\S) semantics: if more input follows (i < n), the greedy
            // match must back off by exactly one whitespace character so that
            // last space stays available to combine with the next token
            // (rule 2's optional non-letter prefix, e.g. " Hello" as one piece).
            // If the run hits end-of-string, consume it in full.
            if (i < n && last_space_start > start){
                i = last_space_start;
            }
            pieces.emplace_back(
                text.substr(start, i - start)
            );
            continue;
        }
        // \s*[\r\n]
        if (cp == '\r' || cp == '\n'){
            i = cur;
            while(i < n){
                size_t old = i;
                uint32_t c = UTF8::decode(text,i);
                if(c == '\r' || c == '\n')
                    continue;
                i = old;
                break;
            }
            pieces.emplace_back(
                text.substr(start,i-start)
            );
            continue;
        }
        // Fallback: single UTF-8 code point safety net
        i = cur;
        pieces.emplace_back(text.substr(start, i - start));
    }
        return pieces;
    }
} // namespace transformer::tokenizer