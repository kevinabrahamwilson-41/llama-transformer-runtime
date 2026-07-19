import os
import torch


# ============================================================
# CONFIG
# ============================================================

WEIGHTS_PATH = (
    "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/"
    "weights/llama_weights.bin"
)

TOKENS = 128
HIDDEN = 2048
INTERMEDIATE = 8192

NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64

EPS = 1e-5

DEVICE = "cpu"


# ============================================================
# DUMP
# ============================================================

def dump_tensor(path, tensor):
    tensor = tensor.detach().float().cpu().reshape(-1)

    with open(path, "w") as file:
        for value in tensor:
            file.write(f"{value.item():.9g}\n")


# ============================================================
# WEIGHT READER
# ============================================================

class WeightReader:

    def __init__(self, path):
        file_size = os.path.getsize(path)
        self.data = torch.from_file(
            path,
            dtype=torch.bfloat16,
            size=file_size // 2
        )

        self.offset = 0

        print(
            f"[PYTORCH] Loaded {self.data.numel()} BF16 elements"
        )

    def read(self, shape, name):

        elements = 1

        for dimension in shape:
            elements *= dimension

        start = self.offset
        end = start + elements

        tensor = self.data[start:end].reshape(shape)

        self.offset = end

        print(
            f"[PYTORCH] {name:<32} "
            f"offset={start:<12} "
            f"elements={elements}"
        )

        return tensor.clone()


# ============================================================
# LOAD LAYER 0
# ============================================================

def load_layer0():

    reader = WeightReader(WEIGHTS_PATH)

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    reader.read(
        (128256, HIDDEN),
        "embedding"
    )

    # --------------------------------------------------------
    # LAYER 0
    #
    # EXACT CUDA CONVERTER ORDER
    # --------------------------------------------------------

    input_norm = reader.read(
        (HIDDEN,),
        "input_layernorm"
    )

    reader.read(
        (HIDDEN, INTERMEDIATE),
        "down_proj"
    )

    reader.read(
        (INTERMEDIATE, HIDDEN),
        "gate_proj"
    )

    reader.read(
        (INTERMEDIATE, HIDDEN),
        "up_proj"
    )

    reader.read(
        (HIDDEN,),
        "post_attention_layernorm"
    )

    k_proj = reader.read(
        (HIDDEN, NUM_KV_HEADS * HEAD_DIM),
        "k_proj"
    )

    reader.read(
        (HIDDEN, HIDDEN),
        "o_proj"
    )

    q_proj = reader.read(
        (HIDDEN, HIDDEN),
        "q_proj"
    )
    
    v_proj = reader.read(
        (HIDDEN, NUM_KV_HEADS * HEAD_DIM),
        "v_proj"
    )

    return (
        input_norm,
        q_proj,
        k_proj,
        v_proj
    )


# ============================================================
# SYNTHETIC INPUT
# ============================================================

def create_input():

    values = [
        1.0 + float(i % 17)
        for i in range(TOKENS * HIDDEN)
    ]

    x = torch.tensor(
        values,
        dtype=torch.float32
    ).reshape(
        TOKENS,
        HIDDEN
    )

    return x.to(torch.bfloat16)


# ============================================================
# RMSNORM
# ============================================================

def rmsnorm(x, weight):

    x_float = x.float()

    variance = (
        x_float * x_float
    ).mean(
        dim=-1,
        keepdim=True
    )

    output = (
        x_float *
        torch.rsqrt(variance + EPS)
    )

    output = output * weight.float()

    return output.to(torch.bfloat16)


# ============================================================
# ROPE
# ============================================================
# ============================================================
# LLAMA 3 ROPE
# ============================================================

def apply_rope(x):

    """
    Input:
        x: [heads, tokens, head_dim]

    Output:
        [heads, tokens, head_dim]
    """

    heads, tokens, head_dim = x.shape

    assert head_dim == HEAD_DIM

    x_float = x.float()

    # --------------------------------------------------------
    # Base inverse frequencies
    # --------------------------------------------------------

    inv_freq = 1.0 / (
        500000.0 **
        (
            torch.arange(
                0,
                head_dim,
                2,
                dtype=torch.float32
            )
            / head_dim
        )
    )

    # --------------------------------------------------------
    # Llama 3 RoPE scaling
    # --------------------------------------------------------

    factor = 32.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context_len = 8192.0

    low_freq_wavelen = (
        old_context_len /
        low_freq_factor
    )

    high_freq_wavelen = (
        old_context_len /
        high_freq_factor
    )

    wavelen = (
        2.0 *
        torch.pi /
        inv_freq
    )
    # --------------------------------------------------------
    # Llama 3 frequency scaling
    # --------------------------------------------------------

    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen,
        inv_freq / factor,
        inv_freq
    )

    smooth_factor = (
        old_context_len / wavelen
        - low_freq_factor
    ) / (
        high_freq_factor
        - low_freq_factor
    )

    smoothed_inv_freq = (
        (1.0 - smooth_factor)
        * inv_freq_llama
        / factor
        +
        smooth_factor
        * inv_freq_llama
    )

    is_medium_freq = (
        (wavelen >= high_freq_wavelen)
        &
        (wavelen <= low_freq_wavelen)
    )

    inv_freq_llama = torch.where(
        is_medium_freq,
        smoothed_inv_freq,
        inv_freq_llama
    )

    # --------------------------------------------------------
    # Position frequencies
    # --------------------------------------------------------

    positions = torch.arange(
        tokens,
        dtype=torch.float32
    )

    freqs = torch.outer(
        positions,
        inv_freq_llama
    )

    # [tokens, head_dim / 2]

    cos = torch.cos(freqs)

    sin = torch.sin(freqs)

    # [1, tokens, head_dim / 2]

    cos = cos.unsqueeze(0)

    sin = sin.unsqueeze(0)

    # --------------------------------------------------------
    # Interleaved pair rotation
    #
    # [x0, x1] -> [x0*c - x1*s,
    #              x0*s + x1*c]
    # --------------------------------------------------------

    x_even = x_float[..., 0::2]

    x_odd = x_float[..., 1::2]

    rotated_even = (
        x_even * cos
        -
        x_odd * sin
    )

    rotated_odd = (
        x_even * sin
        +
        x_odd * cos
    )

    output = torch.empty_like(
        x_float
    )

    output[..., 0::2] = rotated_even

    output[..., 1::2] = rotated_odd

    return output.to(torch.bfloat16)


# ============================================================
# GQA CAUSAL ATTENTION
# ============================================================

def gqa_attention(q, k, v):

    # q: [32, tokens, 64]
    # k: [8,  tokens, 64]
    # v: [8,  tokens, 64]

    q_float = q.float()
    k_float = k.float()
    v_float = v.float()

    output = torch.empty_like(
        q_float
    )

    scale = 1.0 / (HEAD_DIM ** 0.5)

    heads_per_kv = NUM_HEADS // NUM_KV_HEADS

    for head in range(NUM_HEADS):

        kv_head = head // heads_per_kv

        q_head = q_float[head]
        k_head = k_float[kv_head]
        v_head = v_float[kv_head]

        scores = (
            q_head @ k_head.transpose(0, 1)
        ) * scale

        causal_mask = torch.triu(
            torch.ones(
                TOKENS,
                TOKENS,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            causal_mask,
            float("-inf")
        )

        probabilities = torch.softmax(
            scores,
            dim=-1
        )

        output[head] = (
            probabilities @ v_head
        )

    return output.to(torch.bfloat16)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PYTORCH LAYER 0 ATTENTION REFERENCE")
    print("=" * 60)

    print()
    print("[PYTORCH] Loading layer 0 weights...")

    (
        input_norm,
        q_proj,
        k_proj,
        v_proj
    ) = load_layer0()

    print()
    print("[PYTORCH] Creating synthetic input...")

    x = create_input()

    print(
        "[PYTORCH] Input shape:",
        tuple(x.shape)
    )

    # ========================================================
    # RMSNORM
    # ========================================================

    normalized = rmsnorm(
        x,
        input_norm
    )

    dump_tensor(
        "/tmp/pytorch_attention_norm.txt",
        normalized
    )

    # ========================================================
    # Q
    # ========================================================

    q = (
        normalized.float()
        @ q_proj.float()
    ).to(torch.bfloat16)

    dump_tensor(
        "/tmp/pytorch_q.txt",
        q
    )

    # ========================================================
    # K
    # ========================================================

    k = (
        normalized.float()
        @ k_proj.float()
    ).to(torch.bfloat16)

    dump_tensor(
        "/tmp/pytorch_k.txt",
        k
    )

    # ========================================================
    # V
    # ========================================================

    v = (
        normalized.float()
        @ v_proj.float()
    ).to(torch.bfloat16)

    dump_tensor(
        "/tmp/pytorch_v.txt",
        v
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    q = q.reshape(
        TOKENS,
        NUM_HEADS,
        HEAD_DIM
    ).permute(
        1,
        0,
        2
    ).contiguous()

    k = k.reshape(
        TOKENS,
        NUM_KV_HEADS,
        HEAD_DIM
    ).permute(
        1,
        0,
        2
    ).contiguous()

    v = v.reshape(
        TOKENS,
        NUM_KV_HEADS,
        HEAD_DIM
    ).permute(
        1,
        0,
        2
    ).contiguous()

    # ========================================================
    # ROPE
    # ========================================================

    q_rope = apply_rope(q)
    k_rope = apply_rope(k)

    v_rope = v

    dump_tensor(
        "/tmp/pytorch_q_rope.txt",
        q_rope
    )

    dump_tensor(
        "/tmp/pytorch_k_rope.txt",
        k_rope
    )

    dump_tensor(
        "/tmp/pytorch_v_rope.txt",
        v_rope
    )

    # ========================================================
    # GQA ATTENTION
    # ========================================================

    flash_output = gqa_attention(
        q_rope,
        k_rope,
        v_rope
    )

    dump_tensor(
        "/tmp/pytorch_flash_attention.txt",
        flash_output
    )

    # ========================================================
    # LAYOUT BACK
    # ========================================================

    attention_input = (
        flash_output
        .permute(1, 0, 2)
        .contiguous()
        .reshape(TOKENS, HIDDEN)
    )

    dump_tensor(
        "/tmp/pytorch_attention_input.txt",
        attention_input
    )

    print()
    print("[PYTORCH] Reference complete.")
    print()
    print("[PYTORCH] Dumps written to /tmp/pytorch_*.txt")


if __name__ == "__main__":
    main()