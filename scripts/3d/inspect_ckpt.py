"""Quick inspection of a non_lora_trainables.bin produced by train_3d.py.

Usage:
  python scripts/3d/inspect_ckpt.py <ckpt_dir>
  e.g.
  python scripts/3d/inspect_ckpt.py ckpt/llavanext-qwen-video3dllm-sqa3d-lora-jepaonly
"""

import os
import sys

import torch


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ckpt_dir = sys.argv[1]
    bin_path = os.path.join(ckpt_dir, "non_lora_trainables.bin")
    if not os.path.exists(bin_path):
        print(f"[FATAL] not found: {bin_path}")
        sys.exit(2)

    sd = torch.load(bin_path, map_location="cpu")
    keys = list(sd.keys())
    print(f"file: {bin_path}")
    print(f"total keys: {len(keys)}")
    print()
    print("first 20 keys:")
    for k in keys[:20]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else None
        dtype = v.dtype if hasattr(v, "dtype") else None
        finite = "?"
        if hasattr(v, "dtype") and v.is_floating_point():
            finite = bool(torch.isfinite(v).all().item())
        print(f"  {k}  shape={shape} dtype={dtype} finite={finite}")
    print()

    categories = [
        "jepa_projector",
        "embed_tokens",
        "mm_projector",
        "world_position",
        "ground_head",
        "vision_tower",
        "lm_head",
    ]
    print("category breakdown:")
    for cat in categories:
        n = sum(1 for k in keys if cat in k)
        print(f"  {cat:.<28} {n}")

    print()
    print("prefix patterns (first segment of each key):")
    prefixes = {}
    for k in keys:
        first = k.split(".")[0]
        prefixes[first] = prefixes.get(first, 0) + 1
    for p, n in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"  {p:.<28} {n}")


if __name__ == "__main__":
    main()
