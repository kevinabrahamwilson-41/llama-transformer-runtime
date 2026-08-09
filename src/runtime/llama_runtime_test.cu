#include "llama_runtime.hpp"
#include <iostream>
#include <vector>
int main(){
    const char* WEIGHTS =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama_weights.bin";
    const char* TOKENIZER =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/Llama-3.2-1B-Instruct/original/tokenizer.model";
    runtime::LlamaRuntime runtime(
        WEIGHTS,
        TOKENIZER
    );
    std::vector<std::string> prompts ={
        "the capital of USA?"
    };
    for (const auto& prompt : prompts){
        std::cout << "\n=========================================\n";
        std::cout << "Prompt: " << prompt << "\n";
        std::cout << "=========================================\n";
        std::string output = runtime.generate(prompt, 100);
        std::cout << output << ".\n";
    }
    return 0;
}