from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "../../weights/llama-3.2-1b-instruct"

# ------------------------------------------------------------
# Load official tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

# ------------------------------------------------------------
# Load official Llama model
# ------------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True
)

model.eval()

# ------------------------------------------------------------
# EXACT SAME PROMPT
# ------------------------------------------------------------

prompt = (
    "what is the capital of France ?"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to("cuda")

print("=========================================")
print("PROMPT")
print("=========================================")

print(prompt)

print("\nINPUT IDS:")
print(inputs.input_ids)

# ------------------------------------------------------------
# GENERATE
# ------------------------------------------------------------

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        temperature=None,
        top_p=None
    )

# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=False
)

print("\n=========================================")
print("GENERATED OUTPUT")
print("=========================================")

print(generated_text)

print("\n=========================================")
print("GENERATED TOKEN IDS")
print("=========================================")

print(outputs[0])

print("\n=========================================")
print("NEW TOKENS")
print("=========================================")

new_tokens = outputs[0, inputs.input_ids.shape[1]:]

print(new_tokens)

print("\nDECODED NEW TOKENS:")
print(
    tokenizer.decode(
        new_tokens,
        skip_special_tokens=False
    )
)