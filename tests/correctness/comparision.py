import numpy as np
import os

pairs = [
    ("attention_norm", "pytorch_attention_norm.txt", "cuda_attention_norm.txt"),
    ("Q", "pytorch_q.txt", "cuda_q.txt"),
    ("K", "pytorch_k.txt", "cuda_k.txt"),
    ("V", "pytorch_v.txt", "cuda_v.txt"),
    ("Q RoPE", "pytorch_q_rope.txt", "cuda_q_rope.txt"),
    ("K RoPE", "pytorch_k_rope.txt", "cuda_k_rope.txt"),
    ("V RoPE", "pytorch_v_rope.txt", "cuda_v_rope.txt"),
    ("FlashAttention", "pytorch_flash_attention.txt", "cuda_flash_attention.txt"),
    ("Attention Input", "pytorch_attention_input.txt", "cuda_attention_input.txt"),
    ("Attention Output", "pytorch_attention_output.txt", "cuda_attention_output.txt"),
    ("FINAL BLOCK",
 "pytorch_block0_output.txt",
 "cuda_block0_output.txt"),
 # ==========================
    # FFN
    # ==========================

    ("FFN Norm",
     "pytorch_ffn_norm.txt",
     "cuda_ffn_norm.txt"),

    ("Gate Projection",
     "pytorch_gate.txt",
     "cuda_gate.txt"),

    ("Up Projection",
     "pytorch_up.txt",
     "cuda_up.txt"),

    ("SILU(Gate)*Up",
     "pytorch_activated.txt",
     "cuda_activated.txt"),

    ("Down Projection",
     "pytorch_down_proj_output.txt",
     "cuda_down_proj_output.txt")
]

def load(path):
    return np.loadtxt(path, dtype=np.float32)

print("=" * 80)
print("CUDA vs PYTORCH TRANSFORMER BLOCK COMPARISON")
print("=" * 80)

for name, py_file, cuda_file in pairs:
    py_path = "/tmp/" + py_file
    cuda_path = "/tmp/" + cuda_file

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    if not os.path.exists(py_path):
        print("PYTORCH FILE MISSING:", py_path)
        continue

    if not os.path.exists(cuda_path):
        print("CUDA FILE MISSING:", cuda_path)
        continue

    py = load(py_path)
    cuda = load(cuda_path)

    print("PYTORCH elements:", len(py))
    print("CUDA elements   :", len(cuda))

    if len(py) != len(cuda):
        print("❌ SIZE MISMATCH")
        continue

    diff = np.abs(py - cuda)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    index = np.argmax(diff)

    print("max_abs_diff :", max_diff)
    print("mean_abs_diff:", mean_diff)
    print("index        :", index)
    print("PYTORCH      :", py[index])
    print("CUDA         :", cuda[index])

    if max_diff < 0.05:
        print("✅ MATCH")
    else:
        print("❌ DIVERGENCE")
