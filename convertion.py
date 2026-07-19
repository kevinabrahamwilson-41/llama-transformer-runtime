from pathlib import Path
import torch
from safetensors import safe_open
INPUT_PATH = Path(
    "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/"
    "weights/llama-3.2-1b-instruct/model.safetensors"
)

OUTPUT_PATH = Path(
    "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/"
    "weights/llama_weights.bin"
)
NUM_LAYERS = 16
EXPECTED_TENSORS = [
    (
    "model.embed_tokens.weight",
    (128256, 2048),
    ),
]
for layer in range(NUM_LAYERS):
    EXPECTED_TENSORS.extend([
        (
            f"model.layers.{layer}.input_layernorm.weight",
            (2048,),
        ),
        (
            f"model.layers.{layer}.mlp.down_proj.weight",
            (2048, 8192),
        ),

        (
            f"model.layers.{layer}.mlp.gate_proj.weight",
            (8192, 2048),
        ),

        (
            f"model.layers.{layer}.mlp.up_proj.weight",
            (8192, 2048),
        ),

        (
            f"model.layers.{layer}.post_attention_layernorm.weight",
            (2048,),
        ),

        (
            f"model.layers.{layer}.self_attn.k_proj.weight",
            (512, 2048),
        ),

        (
            f"model.layers.{layer}.self_attn.o_proj.weight",
            (2048, 2048),
        ),

        (
            f"model.layers.{layer}.self_attn.q_proj.weight",
            (2048, 2048),
        ),

        (
            f"model.layers.{layer}.self_attn.v_proj.weight",
            (512, 2048),
        ),
    ])

EXPECTED_TENSORS.append(
    (
        "model.norm.weight",
        (2048,),
    )
)

def should_transpose(tensor_name: str) -> bool:
    return (
    tensor_name.endswith(".q_proj.weight")
    or tensor_name.endswith(".k_proj.weight")
    or tensor_name.endswith(".v_proj.weight")
    or tensor_name.endswith(".o_proj.weight")
    or tensor_name.endswith(".gate_proj.weight")
    or tensor_name.endswith(".up_proj.weight")
    or tensor_name.endswith(".down_proj.weight")
    )

def main():
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    print(f"[CONVERT] Input : {input_path}")
    print(f"[CONVERT] Output: {output_path}")
    print()
    with safe_open(
        str(input_path),
        framework="pt",
        device="cpu",
    ) as safetensors_file:
        available_tensors = set(
            safetensors_file.keys()
        )
        with open(
            output_path,
            "wb",
        ) as output_file:
            for index, (tensor_name, expected_shape) in enumerate(
                EXPECTED_TENSORS
            ):
                if tensor_name not in available_tensors:
                    raise RuntimeError(
                        f"Missing tensor: {tensor_name}"
                    )
                tensor = safetensors_file.get_tensor(
                    tensor_name
                )
                if tensor.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"Unexpected dtype for "
                        f"{tensor_name}: "
                        f"{tensor.dtype}"
                    )
                actual_shape = tuple(
                    tensor.shape
                )
                if actual_shape != expected_shape:
                    raise RuntimeError(
                        f"Shape mismatch for "
                        f"{tensor_name}\n"
                        f"Expected: {expected_shape}\n"
                        f"Actual:   {actual_shape}"
                    )
                if should_transpose(tensor_name):
                    tensor = tensor.transpose(
                        0,
                        1,
                    )
                    print(
                        f"[TRANSPOSE] "
                        f"{tensor_name}"
                    )
                tensor = tensor.contiguous()
                raw_bf16 = tensor.view(
                    torch.uint16
                )
                output_file.write(
                    raw_bf16.numpy().tobytes()
                )
                print(
                    f"[{index + 1:03d}/"
                    f"{len(EXPECTED_TENSORS):03d}] "
                    f"{tensor_name:<70} "
                    f"{list(tensor.shape)}"
                )
    print()
    print("[CONVERT] Conversion complete.")
    print(
        f"[CONVERT] Output size: "
        f"{output_path.stat().st_size:,} bytes"
    )
if __name__ == "__main__":
    main()
