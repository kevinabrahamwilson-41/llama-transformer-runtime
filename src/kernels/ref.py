from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
MODEL_PATH = "../../weights/llama-3.2-1b-instruct"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True
)
model.eval()
prompt = "What is 2 + 2?"
print("\n==============================")
print("TOKENIZER TEST")
print("==============================")
tokens = tokenizer(
    prompt,
    return_tensors="pt"
)
print("Input IDs:")
print(tokens.input_ids)
print("\nToken count:")
print(tokens.input_ids.shape)
print("\nDecoded:")
print(
    tokenizer.decode(
        tokens.input_ids[0]
    )
)
# move tokens to GPU
input_ids = tokens.input_ids.cuda()
# ======================================================
# RMSNORM TEST
# ======================================================
print("\n==============================")
print("RMSNORM TEST")
print("==============================")
with torch.no_grad():
    # first token only (matches CUDA forward_next_token)
    token_embedding = model.model.embed_tokens(
        input_ids[:,0:1]
    )
    rms_output = model.model.layers[0].input_layernorm(
        token_embedding
    )
print("RMS output shape:")
print(rms_output.shape)
print("\nFirst token RMS first 32:")
print(
    rms_output[0,0,:32].float()
)
# ======================================================
# Q PROJECTION TEST
# ======================================================
print("\n==============================")
print("WEIGHT ORDER TEST")
print("==============================")

layer = model.model.layers[0]
print("\n==============================")
print("Q WEIGHT LAYOUT TEST")
print("==============================")

W = layer.self_attn.q_proj.weight

print("Original weight shape:")
print(W.shape)

print("\nW[:2,:5]")
print(W[:2,:5].float())

print("\nW.T[:2,:5]")
print(W.T[:2,:5].float())

print("\nW flatten first 10")
print(W.flatten()[:10].float())

print("\nW.T flatten first 10")
print(W.T.flatten()[:10].float())
for name, tensor in [
    ("down", layer.mlp.down_proj.weight),
    ("gate", layer.mlp.gate_proj.weight),
    ("up", layer.mlp.up_proj.weight),
    ("k", layer.self_attn.k_proj.weight),
    ("o", layer.self_attn.o_proj.weight),
    ("q", layer.self_attn.q_proj.weight),
    ("v", layer.self_attn.v_proj.weight)
]:
    print("\n", name)
    print(tensor.flatten()[:10].float())


print("\n==============================")
print("Q PROJECTION TEST")
print("==============================")

print("\nQ WEIGHT TRANSPOSE FIRST 10:")
print(
    layer.self_attn.q_proj.weight.T.flatten()[:10].float()
)

with torch.no_grad():

    # embedding
    hidden_states = model.model.embed_tokens(
        input_ids[:,0:1]
    )

    # attention RMSNorm
    normalized = layer.input_layernorm(
        hidden_states
    )
    print("\n==============================")
    print("MANUAL GEMM TEST")
    print("==============================")

    x = normalized[0,0]

    W = layer.self_attn.q_proj.weight

    q_ref = x @ W.T
    print("\n==============================")
    print("MANUAL GEMM TEST")
    print("==============================")

    x = normalized[0,0].float()
    W = layer.self_attn.q_proj.weight.float()

    print("Normalized first 10:")
    print(x[:10])

    print("\nPyTorch W.T (correct Linear)")
    print((x @ W.T)[:10])

    print("\nPyTorch W (wrong orientation)")
    print((x @ W)[:10])
    print("Normalized first 10:")
    print(x[:10].float())

    print("\nPyTorch manual Q first 32:")
    print(q_ref[:32].float())
    # Q projection
    q = layer.self_attn.q_proj(
        normalized
    )
    print("\nQ WEIGHT T FIRST 10:")
    print(
        model.model.layers[0]
        .self_attn.q_proj.weight.T.flatten()[:10]
        .float()
    )
print("Q shape:")
print(q.shape)

print("\nFirst token Q first 32:")
print(
    q[0,0,:32].float()
)
# ======================================================
# K PROJECTION TEST
# ======================================================

print("\n==============================")
print("K PROJECTION TEST")
print("==============================")

print("\nK WEIGHT ORIGINAL FIRST 10:")
print(
    layer.self_attn.k_proj.weight.flatten()[:10].float()
)

print("\nK WEIGHT TRANSPOSE FIRST 10:")
print(
    layer.self_attn.k_proj.weight.T.flatten()[:10].float()
)

with torch.no_grad():

    k_ref = layer.self_attn.k_proj(
        normalized
    )

print("\nK shape:")
print(k_ref.shape)

print("\nFirst token K first 32:")
print(
    k_ref[0,0,:32].float()
)
# ======================================================
# V PROJECTION TEST
# ======================================================

print("\n==============================")
print("V PROJECTION TEST")
print("==============================")

print("\nV WEIGHT ORIGINAL FIRST 10:")
print(
    layer.self_attn.v_proj.weight.flatten()[:10].float()
)

print("\nV WEIGHT TRANSPOSE FIRST 10:")
print(
    layer.self_attn.v_proj.weight.T.flatten()[:10].float()
)

with torch.no_grad():

    v_ref = layer.self_attn.v_proj(
        normalized
    )

print("\nV shape:")
print(v_ref.shape)

print("\nFirst token V first 32:")
print(
    v_ref[0,0,:32].float()
)
# ======================================================
# ROPE TEST
# ======================================================

print("\n==============================")
print("ROPE TEST")
print("==============================")

with torch.no_grad():

    q = layer.self_attn.q_proj(
        normalized
    )

    k = layer.self_attn.k_proj(
        normalized
    )

    # [batch, seq, hidden]
    q = q.view(
        1,
        1,
        32,
        64
    ).transpose(1,2)

    k = k.view(
        1,
        1,
        8,
        64
    ).transpose(1,2)


    position_ids = torch.tensor(
        [[0]],
        device="cuda"
    )

    cos, sin = model.model.rotary_emb(
        q,
        position_ids
    )

    q_rope, k_rope = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
        unsqueeze_dim=1
    )

print("\nQ ROPE shape:")
print(q_rope.shape)

print("\nQ AFTER ROPE first head first 32:")
print(
    q_rope[0,0,0,:32].float()
)


print("\nK ROPE shape:")
print(k_rope.shape)

print("\nK AFTER ROPE first head first 32:")
print(
    k_rope[0,0,0,:32].float()
)
# ======================================================
# KV CACHE TEST
# ======================================================

print("\n==============================")
print("KV CACHE TEST")
print("==============================")

print("\nK CACHE HEAD0 first 32:")
print(
    k_rope[0,0,0,:32].float()
)

print("\nV CACHE HEAD0 first 32:")
print(
    v_ref.view(1,1,8,64)
    .transpose(1,2)[0,0,0,:32]
    .float()
)
# ======================================================
# ATTENTION TEST
# ======================================================

print("\n==============================")
print("ATTENTION TEST")
print("==============================")

Q = q_rope
K = k_rope

V = (
    v_ref
    .view(1,1,8,64)
    .transpose(1,2)
)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


# repeat KV heads
K = K.repeat_interleave(
    4,
    dim=1
)

V = V.repeat_interleave(
    4,
    dim=1
)


scores = torch.matmul(
    Q,
    K.transpose(-2,-1)
)

scores = scores / (64 ** 0.5)


# causal mask
mask = torch.triu(
    torch.ones_like(scores),
    diagonal=1
)

scores = scores.masked_fill(
    mask.bool(),
    float("-inf")
)


attn = torch.softmax(
    scores,
    dim=-1
)


out = torch.matmul(
    attn,
    V
)


print("\nAttention output shape:")
print(out.shape)


print("\nFLASH OUTPUT HEAD0 first 32:")
print(
    out[0,0,0,:32].float()
)
# ======================================================
# FLASH OUTPUT LAYOUT TEST
# ======================================================

with torch.no_grad():

    # flash output from HF attention
    attn_out = q_rope.transpose(1,2).reshape(
        1,1,2048
    )

print("\n==============================")
print("ATTENTION INPUT REFERENCE")
print("==============================")

print(
    attn_out[0,0,:32].float()
)

# ======================================================
# O PROJECTION INPUT REFERENCE
# ======================================================

with torch.no_grad():

    # [B, heads, seq, head_dim]
    # -> [B, seq, heads*head_dim]

    o_input_ref = (
        out.transpose(1,2)
        .contiguous()
        .reshape(1,1,2048)
    )

print("\nHEAD0")
print(o_input_ref[0,0,0:32].float())

print("\nHEAD1")
print(o_input_ref[0,0,64:96].float())

print("\nHEAD2")
print(o_input_ref[0,0,128:160].float())
print("\n==============================")
print("O PROJECTION INPUT REFERENCE")
print("==============================")

print(
    o_input_ref[0,0,:32].float()
)
print("\n==============================")
print("O INPUT HEAD0")
print("==============================")

print(
    o_input_ref[0,0,0:32].float()
)


print("\n==============================")
print("O INPUT HEAD1")
print("==============================")

print(
    o_input_ref[0,0,64:96].float()
)
# ======================================================
# O PROJECTION REFERENCE
# ======================================================

with torch.no_grad():

    o_ref = layer.self_attn.o_proj(
        o_input_ref
    )

print("\n==============================")
print("O PROJECTION OUTPUT REFERENCE")
print("==============================")

print(
    o_ref[0,0,:32].float()
)