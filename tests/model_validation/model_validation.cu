#include "../../src/runtime/llama_runtime.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main() {
    const std::string weight_path =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama_weights.bin";
    const std::string tokenizer_path =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/Llama-3.2-1B-Instruct/original/tokenizer.model";
    const std::string prompt =
        "Hello, my name is Kevin.";
    constexpr int MAX_NEW_TOKENS = 50;
    try {
        runtime::LlamaRuntime runtime(
            weight_path,
            tokenizer_path
        );
        // --------------------------------------------------------
        // Final logits
        // --------------------------------------------------------
        std::vector<float> logits =
            runtime.generate_logits(prompt);
        std::ofstream logits_file(
            "cuda_logits.txt"
        );
        if(!logits_file){
            throw std::runtime_error(
                "Could not open cuda_logits.txt"
            );
        }
        logits_file.precision(10);
        for(float value : logits){
            logits_file << value << '\n';
        }
        logits_file.close();
        // --------------------------------------------------------
        // Greedy generated token IDs
        // --------------------------------------------------------
        std::vector<int> tokens =
            runtime.generate_token_ids(
                prompt,
                MAX_NEW_TOKENS
            );
        std::ofstream token_file(
            "cuda_tokens.txt"
        );
        if(!token_file){
            throw std::runtime_error(
                "Could not open cuda_tokens.txt"
            );
        }
        for(int token : tokens){
            token_file << token << '\n';
        }
        token_file.close();
        std::cout << "\n";
        std::cout << "CUDA validation complete.\n";
        std::cout << "Prompt: " << prompt << "\n";
        std::cout << "Logits: "
                  << logits.size()
                  << "\n";
        std::cout << "Generated tokens: "
                  << tokens.size()
                  << "\n";

    } catch(const std::exception& e){
        std::cerr
            << "Validation failed: "
            << e.what()
            << "\n";
        return 1;
    }
    return 0;
}