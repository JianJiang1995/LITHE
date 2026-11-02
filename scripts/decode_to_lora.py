#!/usr/bin/env python
"""Decode a .lithe index file into a standard LoRA safetensors file.

The output loads with the stock loader, e.g.
``FluxPipeline.from_pretrained(...).load_lora_weights("lora.safetensors")``.

    python scripts/decode_to_lora.py --lithe adapter.lithe --out lora.safetensors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import lithe

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lithe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    args = ap.parse_args()

    ad = lithe.load_lithe(args.lithe)
    sd = lithe.to_lora_state_dict(ad, dtype=DTYPES[args.dtype])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, args.out, metadata={"format": "pt", "lithe_source": Path(args.lithe).name,
                                      "lithe_rank": str(ad.rank), "lithe_key_format": ad.key_format,
                                      "lithe_metadata": json.dumps(ad.metadata)})
    print(f"{len(ad.modules)} modules -> {len(sd)} tensors, {Path(args.out).stat().st_size / 1e6:.1f} MB "
          f"({args.dtype}) at {args.out}")


if __name__ == "__main__":
    main()
