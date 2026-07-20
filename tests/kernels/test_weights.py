from pathlib import Path
import hashlib
import json


WEIGHTS_PATH = Path(
    "/home/dexter-morgan/PROJECTS/FINAL YEAR PROJECT/"
    "weights/llama_weights.bin"
)

OUTPUT_PATH = Path(
    "/tmp/weights_reference.json"
)


NUM_LAYERS = 16
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
NUM_KV_HEADS = 8
HEAD_DIM = 64
INTERMEDIATE_SIZE = 8192

BF16_BYTES = 2

HASH_CHUNK_BYTES = 16 * 1024 * 1024
PREVIEW_BYTES = 16


def read_tensor(
    file,
    name,
    elements,
    index,
):
    byte_offset = file.tell()
    byte_count = elements * BF16_BYTES

    sha256 = hashlib.sha256()

    first_bytes = bytearray()
    last_bytes = bytearray()

    remaining = byte_count

    while remaining > 0:

        chunk_size = min(
            HASH_CHUNK_BYTES,
            remaining,
        )

        chunk = file.read(chunk_size)

        if len(chunk) != chunk_size:
            raise RuntimeError(
                f"Unexpected EOF while reading {name}"
            )

        sha256.update(chunk)

        if len(first_bytes) < PREVIEW_BYTES:
            needed = PREVIEW_BYTES - len(first_bytes)

            first_bytes.extend(
                chunk[:needed]
            )

        last_bytes = bytearray(
            chunk[-PREVIEW_BYTES:]
        )

        remaining -= chunk_size

    return {
        "index": index,
        "name": name,
        "elements": elements,
        "bytes": byte_count,
        "offset": byte_offset,
        "sha256": sha256.hexdigest(),
        "first_bytes": list(first_bytes),
        "last_bytes": list(last_bytes),
    }


def main():

    print("[PYTHON] Reading:")
    print(f"         {WEIGHTS_PATH}")
    print()

    tensors = []
    index = 0

    with open(WEIGHTS_PATH, "rb") as file:

        # ====================================================
        # EMBEDDING
        # ====================================================

        tensors.append(
            read_tensor(
                file,
                "model.embed_tokens.weight",
                VOCAB_SIZE * HIDDEN_SIZE,
                index,
            )
        )

        index += 1

        # ====================================================
        # LAYERS
        # ====================================================

        for layer in range(NUM_LAYERS):

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.input_layernorm.weight",
                    HIDDEN_SIZE,
                    index,
                )
            )
            index += 1

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.mlp.down_proj.weight",
                    HIDDEN_SIZE * INTERMEDIATE_SIZE,
                    index,
                )
            )
            index += 1

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.mlp.gate_proj.weight",
                    INTERMEDIATE_SIZE * HIDDEN_SIZE,
                    index,
                )
            )
            index += 1

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.mlp.up_proj.weight",
                    INTERMEDIATE_SIZE * HIDDEN_SIZE,
                    index,
                )
            )
            index += 1

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.post_attention_layernorm.weight",
                    HIDDEN_SIZE,
                    index,
                )
            )
            index += 1

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.self_attn.k_proj.weight",
                    HIDDEN_SIZE * NUM_KV_HEADS * HEAD_DIM,
                    index,
                )
            )
            index += 1

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.self_attn.o_proj.weight",
                    HIDDEN_SIZE * HIDDEN_SIZE,
                    index,
                )
            )
            index += 1

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.self_attn.q_proj.weight",
                    HIDDEN_SIZE * HIDDEN_SIZE,
                    index,
                )
            )
            index += 1

            tensors.append(
                read_tensor(
                    file,
                    f"model.layers.{layer}.self_attn.v_proj.weight",
                    HIDDEN_SIZE * NUM_KV_HEADS * HEAD_DIM,
                    index,
                )
            )
            index += 1

        # ====================================================
        # FINAL NORM
        # ====================================================

        tensors.append(
            read_tensor(
                file,
                "model.norm.weight",
                HIDDEN_SIZE,
                index,
            )
        )

        index += 1

        file_size = file.tell()

        trailing = file.read()

        if trailing:
            raise RuntimeError(
                f"Unexpected trailing bytes: {len(trailing)}"
            )

    result = {
        "weights_path": str(WEIGHTS_PATH),
        "file_size": file_size,
        "tensor_count": len(tensors),
        "tensors": tensors,
    }

    with open(OUTPUT_PATH, "w") as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        f"[PYTHON] File size: {file_size:,} bytes"
    )

    print(
        f"[PYTHON] Tensor count: {len(tensors)}"
    )

    print(
        "[PYTHON] Manifest written to:"
    )

    print(
        f"         {OUTPUT_PATH}"
    )

    print()

    for tensor in tensors:

        print(
            f"[{tensor['index']:03d}] "
            f"{tensor['name']:<70} "
            f"offset={tensor['offset']:<12} "
            f"elements={tensor['elements']:<12} "
            f"sha256={tensor['sha256'][:16]}"
        )


if __name__ == "__main__":
    main()