#include "llama_runtime.hpp"
#include <iostream>
#include <vector>

int main()
{
    const char* WEIGHTS =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama_weights.bin";

    const char* TOKENIZER =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama-3.2-1b-instruct/original/tokenizer.model";

    runtime::LlamaRuntime runtime(
        WEIGHTS,
        TOKENIZER
    );

    std::vector<std::string> prompts =
    {

        "What is 2 + 2?",
    };

    for (const auto& prompt : prompts)
    {
        std::cout << "\n=========================================\n";
        std::cout << "Prompt: " << prompt << "\n";
        std::cout << "=========================================\n";

        std::string output = runtime.generate(prompt, 10);

        std::cout << output << "\n";
    }

    return 0;
}