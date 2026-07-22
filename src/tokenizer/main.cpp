#include "tokenizer.hpp"

#include <iostream>

using namespace transformer::tokenizer;

int main()
{
    try
    {
        Tokenizer tokenizer(
            "tokenizer.model"
        );


        std::cout
            << "Vocabulary size: "
            << tokenizer.vocab_size()
            << "\n";


        std::cout
            << "BOS ID: "
            << tokenizer.bos_id()
            << "\n";


        std::cout
            << "EOS ID: "
            << tokenizer.eos_id()
            << "\n";


        std::string text =
            "Hello Llama 3.2";


        auto tokens =
            tokenizer.encode(
                text,
                {
                    .bos=true,
                    .eos=false
                }
            );


        std::cout
            << "\nEncoded tokens:\n";


        for(auto id : tokens)
        {
            std::cout
                << id
                << " ";
        }


        std::cout
            << "\n\nDecoded:\n";


        std::cout
            << tokenizer.decode(tokens)
            << "\n";

    }
    catch(const std::exception& e)
    {
        std::cerr
            << "Tokenizer error: "
            << e.what()
            << "\n";

        return 1;
    }


    return 0;
}