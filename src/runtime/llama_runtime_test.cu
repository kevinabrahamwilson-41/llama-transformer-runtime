#include "llama_runtime.hpp"

#include <iostream>
#include <string>
#include <exception>

int main(){
    const char* WEIGHTS =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/llama_weights.bin";
    const char* TOKENIZER =
        "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/weights/Llama-3.2-1B-Instruct/original/tokenizer.model";
    runtime::LlamaRuntime runtime(
        WEIGHTS,
        TOKENIZER
    );
    std::cout
        << "\n=========================================\n"
        << "       Llama 3.2 Interactive Chat\n"
        << "=========================================\n"
        << "Type /reset to clear conversation\n"
        << "Type /quit to exit\n"
        << "=========================================\n";
    while(true){
        std::string prompt;
        std::cout << "\nYou: ";
        if(!std::getline(std::cin, prompt)){
            break;
        }
        if(prompt == "/quit"){
            break;
        }
        if(prompt == "/reset"){
            runtime.reset_conversation();
            std::cout
                << "Conversation reset.\n";

            continue;
        }
        if(prompt.empty()){
            continue;
        }
        try{
            std::string response =
                runtime.chat(
                    prompt,
                    1000
                );
            std::cout
                << "Assistant: "
                << response
                << "\n";
        }
        catch(const std::exception& e){
            std::cerr
                << "\nInference error: "
                << e.what()
                << "\n";
            break;
        }
    }
    return 0;
}