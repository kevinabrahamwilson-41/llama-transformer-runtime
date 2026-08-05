# Computation Model:

```text
Embedding
    ↓
16 × TransformerBlock
    ↓
Final RMSNorm
    ↓
LM Head / Output Projection
    ↓
Logits
```

## Inside each block:

```text
RMSNorm
    ↓
Q/K/V Linear
    ↓
RoPE
    ↓
KV Cache
    ↓
GQA
    ↓
QKᵀ
    ↓
Scale
    ↓
Causal Mask
    ↓
Softmax
    ↓
Attention × V
    ↓
Output Linear
    ↓
Residual Add
    ↓
RMSNorm
    ↓
W1 ──────┐
         ├─ SiLU → Mul
W3 ──────┘
    ↓
W2
    ↓
Residual Add
```

```
Embedding
    ↓
┌──────────────────────────┐
│ Transformer Block 0      │
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Transformer Block 1      │
└──────────────────────────┘
    ↓
          ...
    ↓
┌──────────────────────────┐
│ Transformer Block 15     │
└──────────────────────────┘
    ↓
Final RMSNorm
    ↓
LM Head
    ↓
Logits
```

```
The reference TransformerBlock explicitly constructs Attention, FeedForward, attention_norm, and ffn_norm.

That's the structural anatomy of ONE block.
```
# MODEL DIMENSIONS
```
hidden_size          = 2048
num_attention_heads  = 32
num_kv_heads         = 8
head_dim             = 64
intermediate_size    = 8192
dtype                = BF16
```
# SINGLE TRANSFORMER BLOCK ANATOMY
```
TransformerBlock
│
├── attention_norm
│   └── RMSNorm(2048)
│
├── attention
│   │
│   ├── wq
│   │   └── 2048 → 2048
│   │
│   ├── wk
│   │   └── 2048 → 512
│   │
│   ├── wv
│   │   └── 2048 → 512
│   │
│   ├── RoPE
│   │
│   ├── KV Cache
│   │
│   ├── GQA
│   │
│   ├── FlashAttention
│   │
│   └── wo
│       └── 2048 → 2048
│
├── ffn_norm
│   └── RMSNorm(2048)
│
└── feed_forward
    │
    ├── w1
    │   └── 2048 → 8192
    │
    ├── w3
    │   └── 2048 → 8192
    │
    ├── SiLU
    │
    ├── Elementwise Mul
    │
    └── w2
        └── 8192 → 2048
```
# Full Transformer Anatomy
```
Transformer
│
├── TransformerBlock[0]
│   ├── Attention
│   │   ├── RMSNorm
│   │   ├── Q/K/V GEMM
│   │   ├── RoPE
│   │   ├── KV Cache
│   │   └── FlashAttention
│   │
│   └── FeedForward
│       ├── RMSNorm
│       ├── W1
│       ├── W3
│       ├── SiLU
│       ├── Mul
│       └── W2
│
├── TransformerBlock[1]
│   ├── Attention
│   └── FeedForward
│
├── ...
│
└── TransformerBlock[15]
    ├── Attention
    └── FeedForward
```
# FeedForward 
```
FeedForward
│
├── GEMM
│   ├── gemm_2048x8192
│   └── gemm_8192x2048
│
├── SiLU
│   └── silu_8192
│
└── Mul
    └── elementwise/mul

```

```
FlashAttention Benchmark

≈137 GFLOPs of attention computation
Execution time: ≈6.6 ms
Implementation: FlashAttention kernel
```

# Runtime Workflow
```
LlamaRuntime                ← whole model inference
│
├── Embedding
│
├── Transformer              ← the 16 decoder layers only
│   │
│   ├── TransformerBlock 0
│   │     ├── Attention
│   │     └── FFN
│   │
│   ├── TransformerBlock 1
│   │
│   ...
│   │
│   └── TransformerBlock 15
│
├── Final RMSNorm
│
├── LM Head
│
└── Sampling
    ├── Argmax
    └── Top-k
```

# BPE Tokenizer
```
Tokenizer
│
├── UTF-8
│   ├── Decode
│   └── Unicode Handling
│
├── Regex Split
│   └── Pre-tokenization
│
├── BPE
│   ├── Initial Symbols
│   ├── Pair Ranking
│   ├── Merge Loop
│   └── Final Tokens
│
├── Vocabulary
│   └── Token → ID
│
└── Special Tokens
    ├── BOS
    ├── EOS
    └── Reserved Tokens
```
# Encoding workflow
```
"Hello world!"
        ↓
Regex Split
        ↓
["Hello", " world", "!"]
        ↓
BPE Encoding
        ↓
["Hello", " world", "!"]
        ↓
Vocabulary Lookup
        ↓
[9906, 1917, 0]
```