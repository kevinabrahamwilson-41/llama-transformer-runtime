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


# ============================================================
# DUMP
# ============================================================

def dump_tensor(path, tensor):

    tensor = (
        tensor
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )

    with open(path, "w") as file:

        for value in tensor:

            file.write(
                f"{value.item():.9g}\n"
            )


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
            f"[PYTORCH] Loaded "
            f"{self.data.numel()} BF16 elements"
        )


    def read(self, shape, name):

        elements = 1

        for dimension in shape:

            elements *= dimension

        start = self.offset

        end = start + elements

        tensor = (
            self.data[start:end]
            .reshape(shape)
        )

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

    reader = WeightReader(
        WEIGHTS_PATH
    )


    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    reader.read(
        (128256, HIDDEN),
        "embedding"
    )


    # --------------------------------------------------------
    # LAYER 0
    # --------------------------------------------------------

    input_norm = reader.read(
        (HIDDEN,),
        "input_layernorm"
    )


    down_proj = reader.read(
        (HIDDEN, INTERMEDIATE),
        "down_proj"
    )


    gate_proj = reader.read(
        (INTERMEDIATE, HIDDEN),
        "gate_proj"
    )


    up_proj = reader.read(
        (INTERMEDIATE, HIDDEN),
        "up_proj"
    )


    post_attention_norm = reader.read(
        (HIDDEN,),
        "post_attention_layernorm"
    )


    k_proj = reader.read(
        (NUM_KV_HEADS * HEAD_DIM, HIDDEN),
        "k_proj"
    )


    o_proj = reader.read(
        (HIDDEN, HIDDEN),
        "o_proj"
    )
    dump_tensor(
        "/tmp/pytorch_o_proj.txt",
        o_proj
    )

    q_proj = reader.read(
        (HIDDEN, HIDDEN),
        "q_proj"
    )


    v_proj = reader.read(
        (NUM_KV_HEADS * HEAD_DIM, HIDDEN),
        "v_proj"
    )


    return (
        input_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        post_attention_norm,
        gate_proj,
        up_proj,
        down_proj
    )


# ============================================================
# SYNTHETIC INPUT
# ============================================================

def create_input():

    values = [

        1.0 + float(i % 17)

        for i in range(
            TOKENS * HIDDEN
        )
    ]

    return torch.tensor(
        values,
        dtype=torch.float32
    ).reshape(
        TOKENS,
        HIDDEN
    ).to(
        torch.bfloat16
    )


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

        x_float
        * torch.rsqrt(
            variance + EPS
        )

    )

    output = (

        output
        * weight.float()
    )

    return output.to(
        torch.bfloat16
    )


# ============================================================
# LLAMA 3 RoPE
# ============================================================

def apply_rope(x):

    # x: [heads, tokens, head_dim]

    heads, tokens, head_dim = x.shape

    assert head_dim == HEAD_DIM

    x_float = x.float()


    inv_freq = 1.0 / (

        500000.0 ** (

            torch.arange(
                0,
                head_dim,
                2,
                dtype=torch.float32
            )
            / head_dim
        )
    )


    factor = 32.0

    low_freq_factor = 1.0

    high_freq_factor = 4.0

    old_context_len = 8192.0


    low_freq_wavelen = (

        old_context_len
        / low_freq_factor
    )


    high_freq_wavelen = (

        old_context_len
        / high_freq_factor
    )


    wavelen = (

        2.0
        * torch.pi
        / inv_freq
    )


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


    positions = torch.arange(
        tokens,
        dtype=torch.float32
    )


    freqs = torch.outer(
        positions,
        inv_freq_llama
    )


    cos = torch.cos(freqs)

    sin = torch.sin(freqs)


    cos = cos.unsqueeze(0)

    sin = sin.unsqueeze(0)


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


    return output.to(
        torch.bfloat16
    )


# ============================================================
# GQA ATTENTION
# ============================================================

def gqa_attention(q, k, v):

    q_float = q.float()

    k_float = k.float()

    v_float = v.float()


    output = torch.empty_like(
        q_float
    )


    scale = 1.0 / (
        HEAD_DIM ** 0.5
    )


    heads_per_kv = (

        NUM_HEADS
        // NUM_KV_HEADS
    )


    causal_mask = torch.triu(

        torch.ones(
            TOKENS,
            TOKENS,
            dtype=torch.bool
        ),

        diagonal=1
    )


    for head in range(NUM_HEADS):

        kv_head = (

            head
            // heads_per_kv
        )


        q_head = q_float[head]

        k_head = k_float[kv_head]

        v_head = v_float[kv_head]


        scores = (

            q_head
            @ k_head.transpose(0, 1)
        ) * scale


        scores = scores.masked_fill(

            causal_mask,

            float("-inf")
        )


        probabilities = torch.softmax(

            scores,

            dim=-1
        )


        output[head] = (

            probabilities
            @ v_head
        )


    return output.to(
        torch.bfloat16
    )


# ============================================================
# ATTENTION
# ============================================================

def attention(

    x,
    input_norm,
    q_proj,
    k_proj,
    v_proj,
    o_proj
):

    normalized = rmsnorm(

        x,

        input_norm
    )


    q = (

        normalized.float()
        @ q_proj.float()
    ).to(
        torch.bfloat16
    )


    k = (

        normalized.float()
        @ k_proj.float().transpose(0, 1)
    ).to(
        torch.bfloat16
    )


    v = (

        normalized.float()
        @ v_proj.float().transpose(0, 1)
    ).to(
        torch.bfloat16
    )


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


    q = apply_rope(q)

    k = apply_rope(k)


    attention_output = gqa_attention(

        q,

        k,

        v
    )
    dump_tensor(
        "/tmp/pytorch_flash_attention_raw.txt",
        attention_output
    )

    print(
        "PY FLASH:",
        attention_output.reshape(-1)[:4]
    )
    print(
        "PY FLASH:",
        attention_output[0, 0, 0].item(),
        attention_output[0, 0, 1].item(),
        attention_output[0, 0, 2].item(),
        attention_output[0, 0, 3].item()
    )

    dump_tensor(
        "/tmp/pytorch_flash_attention.txt",
        attention_output
    )

    attention_input = (

        attention_output
        .permute(1, 0, 2)
        .contiguous()
        .reshape(TOKENS, HIDDEN)
    )
    dump_tensor(
        "/tmp/pytorch_attention_input.txt",
        attention_input
    )

    output = (

        attention_input.float()
        @ o_proj.float().transpose(0, 1)
    ).to(
        torch.bfloat16
    )


    return output


# ============================================================
# FFN
# ============================================================

def feed_forward(

    x,
    post_attention_norm,
    gate_proj,
    up_proj,
    down_proj
):

    normalized = rmsnorm(

        x,

        post_attention_norm
    )


    gate = (

        normalized.float()
        @ gate_proj.float().transpose(0, 1)
    ).to(
        torch.bfloat16
    )


    up = (

        normalized.float()
        @ up_proj.float().transpose(0, 1)
    ).to(
        torch.bfloat16
    )


    gate_float = gate.float()

    up_float = up.float()


    silu = (

        gate_float
        * torch.sigmoid(gate_float)
    )


    activated = (

        silu
        * up_float
    )


    activated = activated.to(
        torch.bfloat16
    )


    output = (

        activated.float()
        @ down_proj.float().transpose(0, 1)
    ).to(
        torch.bfloat16
    )


    return output


# ============================================================
# FULL TRANSFORMER BLOCK
# ============================================================
def transformer_block0():

    (
        input_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        post_attention_norm,
        gate_proj,
        up_proj,
        down_proj

    ) = load_layer0()


    x = create_input()


    attention_output = attention(

        x,

        input_norm,

        q_proj,

        k_proj,

        v_proj,

        o_proj
    )


    dump_tensor(
        "/tmp/pytorch_attention_output.txt",
        attention_output
    )


    residual = (

        x.float()
        + attention_output.float()
    ).to(
        torch.bfloat16
    )


    dump_tensor(
        "/tmp/pytorch_residual.txt",
        residual
    )


    ffn_output = feed_forward(

        residual,

        post_attention_norm,

        gate_proj,

        up_proj,

        down_proj
    )


    dump_tensor(
        "/tmp/pytorch_ffn_output.txt",
        ffn_output
    )


    output = (

        residual.float()
        + ffn_output.float()
    ).to(
        torch.bfloat16
    )


    dump_tensor(
        "/tmp/pytorch_block0_output.txt",
        output
    )


    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "PYTORCH FULL TRANSFORMER BLOCK[0] REFERENCE"
    )

    print("=" * 60)


    print()

    print(
        "[PYTORCH] Running full block..."
    )


    output = transformer_block0()


    dump_tensor(

        "/tmp/pytorch_block0_output.txt",

        output
    )


    print()

    print(

        "[PYTORCH] Reference complete."
    )


    print(

        "[PYTORCH] Dump written to "
        "/tmp/pytorch_block0_output.txt"
    )


if __name__ == "__main__":

    main()
