#include "tokenizer.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>


using namespace transformer::tokenizer;


struct TestCase
{
    std::string name;
    std::string text;
};


int main()
{
    try
    {
        Tokenizer tokenizer(
            "tokenizer.model"
        );


        std::vector<TestCase> tests =
        {

            {
                "basic",
                "Hello Llama 3.2"
            },


            {
                "contractions",
                "I'm I've I'll I'd you're we're they've shouldn't couldn't"
            },


            {
                "punctuation",
                "Hello!!! ??? ... ,,, ;;; ::: --- ___ () [] {} <>"
            },


            {
                "numbers",
                "0 1 12 123 1234 12345 999999999 3.141592653589793"
            },


            {
                "mixed",
                "Llama3.2-1B_Instruct v2.0 CUDA13.0 RTX4060"
            },


            {
                "whitespace",
                "   Hello     world\n\nThis\tis\ta\ttest   "
            },


            {
                "unicode",
                "Café naïve résumé jalapeño München Zürich"
            },


            {
                "emoji",
                "😀 😃 🚀 🔥 🧠 💻 🤖 🌍"
            },


            {
                "unicode_symbols",
                "♥ ★ ✓ ✗ ∑ ∆ ∞ ≈ ≠"
            },


            {
                "asian",
                "你好世界 こんにちは世界 안녕하세요"
            },


            {
                "arabic",
                "مرحبا بالعالم"
            },


            {
                "russian",
                "Привет мир"
            },


            {
                "long_text",
                R"(
The quick brown fox jumps over the lazy dog.
Llama 3.2 is a transformer based language model.
CUDA kernels accelerate tensor operations on NVIDIA GPUs.
The runtime executes optimized BF16 matrix multiplication.
)"
            },


            {
                "code",
                R"(
#include <cuda.h>

__global__ void kernel(float* x)
{
    int idx = threadIdx.x;
    x[idx] *= 2.0f;
}
)"
            },


            {
                "chat_template",
                "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            },


            {
                "special_tokens",
                "<|begin_of_text|> <|end_of_text|> <|eot_id|>"
            },


            {
                "extreme",
                R"(
In 2026, AI systems process 10^12 tokens/day.
The model "Llama-3.2-1B-Instruct" contains 1.23B parameters,
uses Grouped Query Attention (GQA),
supports context windows up to 128k tokens,
and runs inference using custom CUDA kernels,
Tensor Cores, BF16 arithmetic, FlashAttention,
and optimized memory pipelines.

Testing multilingual data:
English 中文 日本語 한국어 العربية Русский Français Español Português.

Emoji stress:
😀🚀🔥🧠💻🤖🌍

Special chars:
!@#$%^&*()_+-=[]{}|;:',.<>/?`~

End.
)"
            }

        };


        for(auto& test : tests)
        {

            std::cout
                << "\n============================\n";

            std::cout
                << test.name
                << "\n";


            auto start =
                std::chrono::high_resolution_clock::now();


            auto tokens =
                tokenizer.encode(
                    test.text,
                    {
                        .bos=true,
                        .eos=false
                    }
                );


            auto end =
                std::chrono::high_resolution_clock::now();


            double us =
                std::chrono::duration<double,std::micro>(
                    end-start
                ).count();



            std::cout
                << "Characters : "
                << test.text.size()
                << "\n";


            std::cout
                << "Tokens     : "
                << tokens.size()
                << "\n";


            std::cout
                << "Time       : "
                << us
                << " us\n";


            std::cout
                << "First IDs  : ";


            for(size_t i=0;i<std::min<size_t>(10,tokens.size());i++)
            {
                std::cout
                    << tokens[i]
                    << " ";
            }

            std::cout
                << "\n";


            auto decoded =
                tokenizer.decode(tokens);


            if(decoded.find(test.text.substr(0,20))
                != std::string::npos)
            {
                std::cout
                    << "Decode sanity: PASS\n";
            }
            else
            {
                std::cout
                    << "Decode sanity: CHECK\n";
            }
        }

        /*
        ============================
        Throughput Benchmark
        ============================
        */


        std::string huge;

        for(int i=0;i<10000;i++)
        {
            huge += tests[16].text;
        }


        std::cout
            << "\n============================\n";

        std::cout
            << "Throughput Benchmark\n";


        auto start =
            std::chrono::high_resolution_clock::now();


        auto benchmark_tokens =
            tokenizer.encode(
                huge,
                {
                    .bos=false,
                    .eos=false
                }
            );


        auto end =
            std::chrono::high_resolution_clock::now();


        double seconds =
            std::chrono::duration<double>(
                end-start
            ).count();


        double mb =
            static_cast<double>(huge.size())
            /
            (1024.0 * 1024.0);


        std::cout
            << "Input size          : "
            << mb
            << " MB\n";


        std::cout
            << "Tokens generated    : "
            << benchmark_tokens.size()
            << "\n";


        std::cout
            << "Time                : "
            << seconds
            << " sec\n";


        std::cout
            << "Throughput          : "
            << mb / seconds
            << " MB/s\n";


        std::cout
            << "Token throughput    : "
            << benchmark_tokens.size()/seconds
            << " tokens/sec\n";


        std::cout
            << "Latency/token       : "
            << (seconds*1'000'000.0)
               /
               benchmark_tokens.size()
            << " us/token\n";

    }   // <-- try closes here
    catch(const std::exception& e)
    {
        std::cerr
            << "FAILED: "
            << e.what()
            << "\n";

        return 1;
    }


    return 0;
}
