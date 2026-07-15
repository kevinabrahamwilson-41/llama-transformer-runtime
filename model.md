
The reference model's actual computation is:

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

Inside each block:

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