#include "utf8.hpp"
namespace transformer::tokenizer{
    uint32_t UTF8::decode(
        std::string_view text,
        size_t& index
    ){
        unsigned char c=
            static_cast<unsigned char>(
                text[index]
            );
        if(c<0x80){
            index++;
            return c;
        }
        if((c>>5)==0x6){
            uint32_t cp=
                ((c&0x1F)<<6) |
                (text[index+1]&0x3F);
            index+=2;
            return cp;
        }
        if((c>>4)==0xE){
            uint32_t cp=
                ((c&0x0F)<<12) |
                ((text[index+1]&0x3F)<<6) |
                (text[index+2]&0x3F);
            index+=3;
            return cp;
        }
        if((c>>3)==0x1E){
            uint32_t cp=
                ((c&0x07)<<18) |
                ((text[index+1]&0x3F)<<12) |
                ((text[index+2]&0x3F)<<6) |
                (text[index+3]&0x3F);
            index+=4;
            return cp;
        }
        index++;
        return 0xFFFD;
    }
    bool UTF8::is_space(uint32_t cp){
        return
            cp == ' '  ||
            cp == '\n' ||
            cp == '\r' ||
            cp == '\t' ||
            cp == '\v' ||   // 0x0B
            cp == '\f' ||   // 0x0C
            cp == 0x0085 || // NEL
            cp == 0x00A0 || // NBSP
            cp == 0x1680 ||
            (cp >= 0x2000 && cp <= 0x200A) || // en quad..hair space
            cp == 0x2028 || // line separator
            cp == 0x2029 || // paragraph separator
            cp == 0x202F || // narrow NBSP
            cp == 0x205F ||
            cp == 0x3000;   // ideographic space
    }
    bool UTF8::is_digit(uint32_t cp){
        if (cp >= '0' && cp <= '9') return true;
        // Common non-ASCII decimal digit blocks
        if (cp >= 0x0660 && cp <= 0x0669) return true; // Arabic-Indic
        if (cp >= 0x06F0 && cp <= 0x06F9) return true; // Extended Arabic-Indic
        if (cp >= 0x0966 && cp <= 0x096F) return true; // Devanagari
        if (cp >= 0x09E6 && cp <= 0x09EF) return true; // Bengali
        if (cp >= 0xFF10 && cp <= 0xFF19) return true; // Fullwidth digits

        return false;
    }
    bool UTF8::is_letter(uint32_t cp){
        // ASCII
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'))
            return true;
        // Latin-1 Supplement letters (skip × U+00D7 and ÷ U+00F7, which are symbols)
        if (cp >= 0x00C0 && cp <= 0x02AF) {
            if (cp == 0x00D7 || cp == 0x00F7) return false;
            return true;
        }
        // Combining diacritical marks (Mn category, but \p{L} contexts often need
        // these to stay attached to the preceding letter for correct BPE runs)
        if (cp >= 0x0300 && cp <= 0x036F) return true;
        // Greek and Coptic
        if (cp >= 0x0370 && cp <= 0x03FF) return true;
        // Cyrillic + Cyrillic Supplement
        if (cp >= 0x0400 && cp <= 0x052F) return true;
        // Armenian
        if (cp >= 0x0530 && cp <= 0x058F) return true;
        // Hebrew
        if (cp >= 0x0590 && cp <= 0x05FF) return true;
        // Arabic + Arabic Supplement
        if (cp >= 0x0600 && cp <= 0x06FF) return true;
        if (cp >= 0x0750 && cp <= 0x077F) return true;
        // Devanagari through Sinhala (Indic scripts block)
        if (cp >= 0x0900 && cp <= 0x0DFF) return true;
        // Thai
        if (cp >= 0x0E00 && cp <= 0x0E7F) return true;
        // Georgian
        if (cp >= 0x10A0 && cp <= 0x10FF) return true;
        // Hangul Jamo
        if (cp >= 0x1100 && cp <= 0x11FF) return true;
        // Latin Extended Additional
        if (cp >= 0x1E00 && cp <= 0x1EFF) return true;
        // CJK Unified Ideographs (+ Extension A)
        if (cp >= 0x3400 && cp <= 0x4DBF) return true;
        if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
        // Hiragana + Katakana
        if (cp >= 0x3040 && cp <= 0x30FF) return true;
        // Hangul Syllables
        if (cp >= 0xAC00 && cp <= 0xD7A3) return true;
        // CJK Compatibility Ideographs
        if (cp >= 0xF900 && cp <= 0xFAFF) return true;
        // CJK Extension B+ (astral plane, needs your decode() to actually produce
        // codepoints this high — check that 4-byte UTF-8 decoding is correct)
        if (cp >= 0x20000 && cp <= 0x2FA1F) return true;
        return false;
    }
}