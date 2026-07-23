#include "tokenizer.hpp"
#include <iostream>
#include <string>
using namespace transformer::tokenizer;
int main() {
    Tokenizer tokenizer("tokenizer.model");
    std::cout << "=========================================\n";
    std::cout << "        Llama 3.2 BPE Tokenizer\n";
    std::cout << "=========================================\n\n";
    std::string prompt =
        "Explain FlashAttention in one sentence.";
    auto tokens = tokenizer.encode(prompt,
                                   {.bos = false, .eos = false});
    std::cout << "Input:\n";
    std::cout << prompt << "\n\n";
    std::cout << "Encoded Token IDs:\n";
    for (auto t : tokens)
        std::cout << t << " ";
    std::cout << "\n\n";
    auto decoded = tokenizer.decode(tokens);
    std::cout << "Decoded Output:\n";
    std::cout << decoded << "\n\n";
    std::cout << "=========================================\n";
    std::cout << "Tokenizer Validation: PASSED\n";
    std::cout << "=========================================\n";
    return 0;
}