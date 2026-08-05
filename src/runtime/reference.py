from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_PATH = "../../weights/llama-3.2-1b-instruct"


# ==================================================
# LOAD TOKENIZER
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


# ==================================================
# LOAD MODEL
# ==================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True,
    attn_implementation="eager"
)

model.eval()


# ==================================================
# MATCH CUDA INPUT
# ==================================================

token_id = 128000

input_ids = torch.tensor(
    [[token_id]],
    dtype=torch.long,
    device="cuda"
)


# ==================================================
# FORWARD
# ==================================================
# ==================================================
# CAPTURE REAL ATTENTION OUTPUT
# ==================================================

attn_output_capture = {}

def attn_hook(module, input, output):
    print("\n==============================")
    print("SELF ATTENTION HOOK")
    print("==============================")

    print("OUTPUT TYPE:")
    print(type(output))

    if isinstance(output, tuple):
        print("TUPLE LENGTH:", len(output))

        for i, x in enumerate(output):
            if torch.is_tensor(x):
                print(
                    f"OUTPUT[{i}] SHAPE:",
                    x.shape
                )

        attn_output_capture["out"] = output[0]

    else:
        print("SHAPE:", output.shape)
        attn_output_capture["out"] = output


handle = model.model.layers[0].self_attn.register_forward_hook(
    attn_hook
)
qkv_capture = {}
def qkv_hook(module, args, kwargs, output):

    hidden_states = kwargs["hidden_states"]

    print("\nSELF ATTENTION INPUT")
    print(hidden_states.shape)


    rms_weight = model.model.layers[0].input_layernorm.weight
    print("\n==============================")
    print("PYTHON RMS INPUT FIRST 32")
    print("==============================")
    print(
        hidden_states[0,0,:32].float()
    )
    rms_out = torch.nn.functional.rms_norm(
        hidden_states,
        (2048,),
        weight=rms_weight,
        eps=1e-5
    )


    print("\n==============================")
    print("PYTHON RMSNORM OUTPUT FIRST 32")
    print("==============================")

    print(
        rms_out[0,0,:32].float()
    )


    k_proj = module.k_proj(rms_out)

    print("\nPYTHON RAW K PROJECTION HEAD0 FIRST 32")

    k = k_proj.reshape(
        1,
        1,
        8,
        64
    )

    print(
        k[0,0,0,:32].float()
    )
handle2 = model.model.layers[0].self_attn.register_forward_hook(
    qkv_hook,
    with_kwargs=True
)
with torch.no_grad():

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    )


# ==================================================
# ATTENTION DEBUG
# ==================================================

print("\n==============================")
print("ATTENTION DEBUG")
print("==============================")


print(
    "NUMBER OF LAYERS:",
    len(outputs.attentions)
)


for i, attn in enumerate(outputs.attentions):

    print(
        f"LAYER {i} ATTENTION SHAPE:",
        attn.shape
    )


# ==================================================
# LAYER 0 ATTENTION
# ==================================================

attn0 = outputs.attentions[0]


print("\n==============================")
print("LAYER 0 HEAD 0 ATTENTION")
print("==============================")


print(
    "Attention shape:",
    attn0.shape
)


print(
    attn0[0,0,0,:].float()
)


# ==================================================
# KV CACHE DEBUG
# ==================================================

print("\n==============================")
print("KV CACHE DEBUG")
print("==============================")


past = outputs.past_key_values


print("CACHE TYPE:")
print(type(past))


print("\nCACHE DIR:")
print(dir(past))


print("\nNUMBER OF CACHE LAYERS:")
print(len(past.layers))


# layer 0
layer0 = past.layers[0]


print("\nLAYER0 TYPE:")
print(type(layer0))


print("\nLAYER0 DIR:")
print(dir(layer0))


# New transformers cache API
layer0_k = layer0.keys
layer0_v = layer0.values


print("\nK SHAPE:")
print(layer0_k.shape)


print("\nV SHAPE:")
print(layer0_v.shape)



print("\nLayer0 K HEAD0 TOKEN0 FIRST 32")

print(
    layer0_k[0,0,0,:32].float()
)



print("\nLayer0 V HEAD0 TOKEN0 FIRST 32")

print(
    layer0_v[0,0,0,:32].float()
)

# ==================================================
# BEFORE FINAL RMSNorm
# ==================================================

hidden = outputs.hidden_states[-1][0,0]


print("\n==============================")
print("FINAL HIDDEN BEFORE RMSNorm")
print("==============================")


print(
    hidden[:32].float()
)


torch.set_printoptions(
    precision=9,
    linewidth=200
)


with open("/tmp/python_before_final_norm.txt","w") as f:

    for x in hidden.float():

        f.write(
            f"{x.item():.9f}\n"
        )



# ==================================================
# AFTER FINAL RMSNorm
# ==================================================

normed = model.model.norm(
    outputs.hidden_states[-1]
)[0,0]


print("\n==============================")
print("FINAL HIDDEN AFTER RMSNorm")
print("==============================")


print(
    normed[:32].float()
)



with open("/tmp/python_final_norm.txt","w") as f:

    for x in normed.float():

        f.write(
            f"{x.item():.9f}\n"
        )



# ==================================================
# LM HEAD
# ==================================================

logits = model.lm_head(normed)


print("\n==============================")
print("LOGITS FIRST 10")
print("==============================")


print(
    logits[:10].float()
)



# ==================================================
# ARGMAX
# ==================================================

print("\n==============================")
print("ARGMAX")
print("==============================")


print(
    torch.argmax(logits).item()
)