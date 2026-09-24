#include "llama_runtime.hpp"

#include <iostream>
#include <string>
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

    // Sequence lengths to benchmark
    const std::vector<int> seq_lengths = {
        32,
        64,
        128,
        256,
        512,
        1024,
        2048
    };

    for (int seq_len : seq_lengths) {

        std::cout
            << "\n\n=========================================\n"
            << "       SEQUENCE LENGTH: " << seq_len << "\n"
            << "=========================================\n";

        // Clear previous KV cache / conversation
        runtime.reset_conversation();

        // Temporary scaling prompt.
        // The runtime itself will report the actual token count.
        std::string prompt;

        for (int i = 0; i < seq_len; ++i) {
            prompt += "hello ";
        }

        try {

            // Generate a fixed number of tokens.
            // Your runtime will print all performance statistics.
            runtime.chat(
                prompt,
                100
            );

        }
        catch (const std::exception& e) {

            std::cerr
                << "\nInference error: "
                << e.what()
                << "\n";

            return 1;
        }
    }

    return 0;
}