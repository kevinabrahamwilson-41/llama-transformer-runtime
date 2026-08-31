```
                 Llama Transformer
                        │
                        ▼
                    RMSNorm
                        │
                        ▼
                 Q / K / V GEMMs
                        │
                        ▼
                     RoPE
                 rotates Q and K
                        │
                        ▼
               ┌─────────────────┐
               │ Grouped Query   │
               │ Attention       │
               │                 │
               │ 32 Q heads      │
               │  8 KV heads     │
               └─────────────────┘
                        │
                        ▼
                   KV Cache
              stores previous K,V
                        │
                        ▼
                    Attention
                        │
                        ▼
                  Output / Residual
                        │
                        ▼
                    RMSNorm
                        │
                        ▼
                    SwiGLU
               ┌────────────────┐
               │ Up/Gate GEMMs  │
               │ SiLU × Gate    │
               │ Down GEMM      │
               └────────────────┘
                        │
                        ▼
                Transformer output
                        │
                        ▼
                 Next-token logits
                        │
                        ▼
              Autoregressive generation
                        │
                        ▼
                   Next token
```