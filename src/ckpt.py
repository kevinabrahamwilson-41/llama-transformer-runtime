import struct
import numpy as np
import torch
from pathlib import Path

CUDA = Path("cuda_debug")
PY = Path("kernels/python_debug")


# ============================================================
# CUDA CHECKPOINT LOADER
#
# Header:
#   uint64 count
#   uint32 dtype
#   uint32 reserved
#   float32 data[count]
# ============================================================

def load_cuda(path):
    with open(path, "rb") as f:

        count = struct.unpack("<Q", f.read(8))[0]
        dtype = struct.unpack("<I", f.read(4))[0]
        reserved = struct.unpack("<I", f.read(4))[0]

        if dtype != 0:
            raise RuntimeError(
                f"Unsupported CUDA dtype code: {dtype}"
            )

        x = np.fromfile(
            f,
            dtype=np.float32,
            count=count
        )

    if x.size != count:
        raise RuntimeError(
            f"Expected {count} FP32 values, "
            f"but file contains {x.size}"
        )

    return x


# ============================================================
# PYTORCH CHECKPOINT LOADER
#
# {
#   "name": ...,
#   "shape": ...,
#   "dtype": ...,
#   "data": Tensor
# }
# ============================================================

def load_pt(path):

    x = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(x, dict):
        raise RuntimeError(
            f"Expected dict checkpoint, got {type(x)}"
        )

    if "data" not in x:
        raise RuntimeError(
            f"Checkpoint has no 'data' field. "
            f"Keys: {list(x.keys())}"
        )

    tensor = x["data"]

    return (
        tensor
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )


# ============================================================
# CHECKPOINT PAIRS
# ============================================================

pairs = [

    ("embedding.bin",                  "embedding.pt"),

    ("layer0_rms1_input.bin",          "layer0_rms1_input.pt"),
    ("layer0_rms1_output.bin",         "layer0_rms1_output.pt"),

    ("layer0_q_projection.bin",        "layer0_q_projection.pt"),
    ("layer0_q_rope.bin",              "layer0_q_rope.pt"),

    ("layer0_k_projection.bin",        "layer0_k_projection.pt"),
    ("layer0_k_rope.bin",              "layer0_k_rope.pt"),

    ("layer0_v_projection.bin",        "layer0_v_projection.pt"),

    ("flash_q_input.bin",              "layer0_q_heads.pt"),
    ("flash_k_input.bin",              "layer0_k_gqa.pt"),
    ("flash_v_input.bin",              "layer0_v_gqa.pt"),

    ("layer0_attention_merged.bin",    "layer0_attention_merged.pt"),

    ("layer0_attention_output_heads.bin",
                                       "layer0_attention_output_heads.pt"),

    ("layer0_o_projection_input.bin",  "layer0_o_projection_input.pt"),
    ("layer0_o_projection_output.bin", "layer0_o_projection_output.pt"),

    ("layer0_attention_residual.bin",  "layer0_attention_residual.pt"),

    ("layer0_rms2_input.bin",          "layer0_rms2_input.pt"),
    ("layer0_rms2_output.bin",         "layer0_rms2_output.pt"),

    ("layer0_output.bin",              "layer0_output.pt"),
]


# ============================================================
# COMPARISON
# ============================================================

for cuda_name, py_name in pairs:

    cuda_path = CUDA / cuda_name
    py_path = PY / py_name

    print("\n" + "=" * 79)
    print(f"{cuda_name}  <->  {py_name}")

    if not cuda_path.exists():
        print("CUDA FILE MISSING")
        continue

    if not py_path.exists():
        print("PYTHON FILE MISSING")
        continue

    try:
        a = load_cuda(cuda_path)
        b = load_pt(py_path)

        # Also load metadata for shape reporting
        pt_meta = torch.load(
            py_path,
            map_location="cpu",
            weights_only=True
        )

    except Exception as e:
        print("LOAD ERROR:", e)
        continue

    print(f"CUDA elements : {a.size}")
    print(f"PY elements   : {b.size}")

    if isinstance(pt_meta, dict):
        print(f"PY shape      : {pt_meta.get('shape')}")
        print(f"PY dtype      : {pt_meta.get('dtype')}")

    if a.size != b.size:
        print(">>> SIZE MISMATCH <<<")
        continue

    diff = np.abs(a - b)

    max_idx = int(np.argmax(diff))
    max_diff = float(diff[max_idx])
    mean_diff = float(np.mean(diff))

    bad_001 = int(np.sum(diff > 0.01))
    bad_01 = int(np.sum(diff > 0.1))

    print()
    print(f"max diff      : {max_diff:.10f}")
    print(f"mean diff     : {mean_diff:.10f}")
    print(f"> 0.01        : {bad_001}")
    print(f"> 0.1         : {bad_01}")
    print(f"max index     : {max_idx}")

    print("\nFIRST 10:")

    for i in range(min(10, a.size)):
        print(
            f"{i:5d}: "
            f"CUDA={a[i]: .8f} "
            f"PY={b[i]: .8f} "
            f"DIFF={diff[i]: .8f}"
        )

    if max_diff < 1e-3:
        print("\n>>> MATCH <<<")
    else:
        print("\n>>> MISMATCH <<<")

    # --------------------------------------------------------
    # First significant mismatch
    # --------------------------------------------------------

    mismatch = np.where(diff > 0.01)[0]

    if mismatch.size > 0:

        idx = int(mismatch[0])

        print("\nFIRST > 0.01 MISMATCH:")
        print(
            f"index {idx}: "
            f"CUDA={a[idx]:.8f} "
            f"PY={b[idx]:.8f} "
            f"DIFF={diff[idx]:.8f}"
        )

    print()
