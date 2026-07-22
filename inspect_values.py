import json
import struct
PATH = "weights/llama-3.2-1b-instruct/model.safetensors"

with open(PATH, "rb") as f:
    # First 8 bytes = header size
    header_size = struct.unpack("<Q", f.read(8))[0]

    # Read JSON header
    header_bytes = f.read(header_size)
    header = json.loads(header_bytes)

for key in sorted(header.keys()):
    if key == "__metadata__":
        continue

    info = header[key]

    print("=" * 100)
    print(f"NAME : {key}")
    print(f"DTYPE: {info['dtype']}")
    print(f"SHAPE: {info['shape']}")
    print(f"OFFSETS: {info['data_offsets']}")