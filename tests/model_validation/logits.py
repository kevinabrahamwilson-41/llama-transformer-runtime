from pyexpat import model

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = (
    "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/"
    "weights/Llama-3.2-1B-Instruct"
)

PROMPT = "Hello, my name is Kevin."
MAX_NEW_TOKENS = 50

CUDA_LOGITS = "cuda_logits.txt"
CUDA_TOKENS = "cuda_tokens.txt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    # ------------------------------------------------------------
    # Load reference model
    # ------------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    model.eval()

    # ------------------------------------------------------------
    # Tokenize prompt
    # ------------------------------------------------------------

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)

    print("Reference input IDs:")
    print(input_ids[0].tolist())

    # ------------------------------------------------------------
    # Reference logits
    # ------------------------------------------------------------

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids
        )

    reference_logits = (
        outputs.logits[0, -1]
        .float()
        .cpu()
        .numpy()
    )

    # ------------------------------------------------------------
    # Load CUDA logits
    # ------------------------------------------------------------

    cuda_logits = np.loadtxt(
        CUDA_LOGITS,
        dtype=np.float32
    )

    print()
    print(f"CUDA logits shape:      {cuda_logits.shape}")
    print(f"Reference logits shape: {reference_logits.shape}")

    if cuda_logits.shape != reference_logits.shape:
        raise RuntimeError(
            "Logit shape mismatch"
        )

    # ------------------------------------------------------------
    # Maximum absolute error
    # ------------------------------------------------------------

    max_abs_error = np.max(
        np.abs(
            cuda_logits -
            reference_logits
        )
    )

    # ------------------------------------------------------------
    # Mean absolute error
    # ------------------------------------------------------------

    mean_abs_error = np.mean(
        np.abs(
            cuda_logits -
            reference_logits
        )
    )

    # ------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------

    dot = np.dot(
        cuda_logits,
        reference_logits
    )

    cuda_norm = np.linalg.norm(
        cuda_logits
    )

    reference_norm = np.linalg.norm(
        reference_logits
    )

    cosine_similarity = (
        dot /
        (cuda_norm * reference_norm)
    )

# ------------------------------------------------------------
    # Reference greedy generation with KV cache
    # ------------------------------------------------------------

    reference_tokens = []

    with torch.no_grad():

        # Prefill: process the complete prompt once
        outputs = model(
            input_ids=input_ids,
            use_cache=True
        )

        # First generated token
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1
        )

        past_key_values = outputs.past_key_values

        for _ in range(MAX_NEW_TOKENS):

            token_id = int(
                next_token.item()
            )

            reference_tokens.append(
                token_id
            )

            # Decode: process only the newly generated token
            outputs = model(
                input_ids=next_token[:, None],
                past_key_values=past_key_values,
                use_cache=True
            )

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1
            )

            past_key_values = outputs.past_key_values

    print("Reference generated tokens:")

    for i, token in enumerate(reference_tokens[:10]):
        print(f"{i}: {token}")
    # ------------------------------------------------------------
    # Load CUDA generated tokens
    # ------------------------------------------------------------

    cuda_tokens = np.loadtxt(
        CUDA_TOKENS,
        dtype=np.int64
    )

    cuda_tokens = np.atleast_1d(
        cuda_tokens
    )

    reference_tokens = np.asarray(
        reference_tokens,
        dtype=np.int64
    )

    count = min(
        len(cuda_tokens),
        len(reference_tokens)
    )

    token_matches = (
        cuda_tokens[:count] ==
        reference_tokens[:count]
    )

    matching_tokens = int(
        np.sum(token_matches)
    )

    token_agreement = (
        matching_tokens / count
    )

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print()
    print("========================================")
    print("MODEL OUTPUT VALIDATION")
    print("========================================")

    print(f"Prompt: {PROMPT}")
    print()

    print(
        f"Maximum absolute logit error: "
        f"{max_abs_error:.8e}"
    )

    print(
        f"Mean absolute logit error:    "
        f"{mean_abs_error:.8e}"
    )

    print(
        f"Cosine similarity:             "
        f"{cosine_similarity:.10f}"
    )

    print()

    print(
        f"CUDA generated tokens:         "
        f"{len(cuda_tokens)}"
    )

    print(
        f"Reference generated tokens:    "
        f"{len(reference_tokens)}"
    )

    print(
        f"Compared tokens:               "
        f"{count}"
    )

    print(
        f"Matching tokens:               "
        f"{matching_tokens}"
    )

    print(
        f"Token agreement:               "
        f"{token_agreement * 100.0:.2f}%"
    )

    print("========================================")


if __name__ == "__main__":
    main()
