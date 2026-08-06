from transformers import AutoModelForCausalLM
import torch

MODEL_PATH = "../../weights/llama-3.2-1b-instruct"


# ==================================================
# LOAD MODEL
# ==================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True
)

model.eval()


layer0 = model.model.layers[0]


# ==================================================
# Q WEIGHT DEBUG
# ==================================================

q_weight = layer0.self_attn.q_proj.weight


print("\n==============================")
print("PYTHON Q WEIGHT SHAPE")
print("==============================")

print(q_weight.shape)


print("\n==============================")
print("PYTHON Q WEIGHT ROW 0 FIRST 10")
print("==============================")

print(
    q_weight[0, :10].float()
)


print("\n==============================")
print("PYTHON Q WEIGHT COLUMN 0 FIRST 10")
print("==============================")

print(
    q_weight[:, 0][:10].float()
)



# ==================================================
# INPUT TOKEN
# ==================================================

token_id = 128000


input_ids = torch.tensor(
    [[token_id]],
    dtype=torch.long,
    device="cuda"
)



# ==================================================
# EMBEDDING
# ==================================================

with torch.no_grad():

    embedding = model.model.embed_tokens(
        input_ids
    )


print("\n==============================")
print("PYTHON EMBEDDING OUTPUT")
print("==============================")

print(
    embedding[0,0,:32].float()
)


# ==================================================
# RMSNORM
# ==================================================

with torch.no_grad():

    rms_weight = (
        layer0.input_layernorm.weight
    )

    rms_out = torch.nn.functional.rms_norm(
        embedding,
        (2048,),
        weight=rms_weight,
        eps=1e-5
    )


print("\n==============================")
print("PYTHON RMS OUTPUT")
print("==============================")

print(
    rms_out[0,0,:32].float()
)



# ==================================================
# Q PROJECTION
# ==================================================

with torch.no_grad():

    q_out = layer0.self_attn.q_proj(
        rms_out
    )


print("\n==============================")
print("PYTHON Q OUTPUT")
print("==============================")

print(
    q_out[0,0,:32].float()
)



# ==================================================
# SAVE FILES FOR COMPARISON
# ==================================================

with open(
    "/tmp/python_embedding.txt",
    "w"
) as f:

    for x in embedding[0,0].float():
        f.write(
            f"{x.item():.9f}\n"
        )



with open(
    "/tmp/python_rms.txt",
    "w"
) as f:

    for x in rms_out[0,0].float():
        f.write(
            f"{x.item():.9f}\n"
        )



with open(
    "/tmp/python_q_output.txt",
    "w"
) as f:

    for x in q_out[0,0].float():
        f.write(
            f"{x.item():.9f}\n"
        )


print("\nSaved:")
print("/tmp/python_embedding.txt")
print("/tmp/python_rms.txt")
print("/tmp/python_q_output.txt")