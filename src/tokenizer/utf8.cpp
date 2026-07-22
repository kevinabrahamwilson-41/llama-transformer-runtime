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
        cp==' ' ||
        cp=='\n' ||
        cp=='\r' ||
        cp=='\t';

}



bool UTF8::is_digit(uint32_t cp){

    return
        cp>='0' &&
        cp<='9';

}



bool UTF8::is_letter(uint32_t cp){

    /*
        Basic Latin letters.
        We will extend this with Unicode ranges
        when needed.
    */

    return
        (cp>='A'&&cp<='Z') ||
        (cp>='a'&&cp<='z');

}


}