from transformers import AutoTokenizer
import time
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct"
)
class TestCase:
    def __init__(self,name,text):
        self.name=name
        self.text=text
tests = [
    TestCase(
    "basic",
    "Hello Llama 3.2"
    ),
    TestCase(
    "contractions",
    "I'm I've I'll I'd you're we're they've shouldn't couldn't"
    ),
    TestCase(
    "punctuation",
    "Hello!!! ??? ... ,,, ;;; ::: --- ___ () [] {} <>"
    ),
    TestCase(
    "numbers",
    "0 1 12 123 1234 12345 999999999 3.141592653589793"
    ),
    TestCase(
    "mixed",
    "Llama3.2-1B_Instruct v2.0 CUDA13.0 RTX4060"
    ),
    TestCase(
    "whitespace",
    "   Hello     world\n\nThis\tis\ta\ttest   "
    ),
    TestCase(
    "unicode",
    "Café naïve résumé jalapeño München Zürich"
    ),
    TestCase(
    "emoji",
    "😀 😃 🚀 🔥 🧠 💻 🤖 🌍"
    ),
    TestCase(
    "unicode_symbols",
    "♥ ★ ✓ ✗ ∑ ∆ ∞ ≈ ≠"
    ),
    TestCase(
    "asian",
    "你好世界 こんにちは世界 안녕하세요"
    ),
    TestCase(
    "arabic",
    "مرحبا بالعالم"
    ),
    TestCase(
    "russian",
    "Привет мир"
    ),
    TestCase(
    "long_text",
    """
    The quick brown fox jumps over the lazy dog.
    Llama 3.2 is a transformer based language model.
    CUDA kernels accelerate tensor operations on NVIDIA GPUs.
    The runtime executes optimized BF16 matrix multiplication.
    """
    ),
    TestCase(
    "code",
    """
    #include <cuda.h>

    __global__ void kernel(float* x)
    {
        int idx = threadIdx.x;
        x[idx] *= 2.0f;
    }
    """
    ),
    TestCase(
    "chat_template",
    "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
    ),
    TestCase(
    "special_tokens",
    "<|begin_of_text|> <|end_of_text|> <|eot_id|>"
    ),
    TestCase(
    "extreme",
    """
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
    """
    )
]
print("\n============================")
print("Python HuggingFace Tokenizer")
print("============================")
for test in tests:
    start=time.perf_counter()
    ids=tokenizer.encode(
        test.text,
        add_special_tokens=True
    )
    end=time.perf_counter()
    us=(end-start)*1_000_000
    print("\n============================")
    print(test.name)
    print(
        "Characters :",
        len(test.text.encode("utf-8"))
    )
    print(
        "Tokens     :",
        len(ids)
    )
    print(
        "Time       :",
        us,
        "us"
    )
    print(
        "First IDs  :",
        ids[:10]
    )
# ===========================
# Throughput benchmark
# ===========================
huge=""
for i in range(10000):
    huge += tests[16].text
print("\n============================")
print("Throughput Benchmark")
print("============================")
start=time.perf_counter()
ids=tokenizer.encode(
    huge,
    add_special_tokens=False
)
end=time.perf_counter()
seconds=end-start
mb=len(huge.encode("utf-8"))/(1024*1024)
print(
    "Input size          :",
    mb,
    "MB"
)
print(
    "Tokens generated    :",
    len(ids)
)
print(
    "Time                :",
    seconds,
    "sec"
)
print(
    "Throughput          :",
    mb/seconds,
    "MB/s"
)
print(
    "Token throughput    :",
    len(ids)/seconds,
    "tokens/sec"
)
print(
    "Latency/token       :",
    seconds*1_000_000/len(ids),
    "us/token"
)