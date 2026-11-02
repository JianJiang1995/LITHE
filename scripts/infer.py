#!/usr/bin/env python
"""Generate images on FLUX.1-dev with a LITHE index file.

  --mode peft    decode the indices to a dense LoRA and load it with the stock
                 ``FluxPipeline.load_lora_weights`` (default)
  --mode native  wrap the target linears with LitheLinear and decode the factors
                 from the index streams directly (diffusers-named adapters only)

    python scripts/infer.py --pretrained ./checkpoints/FLUX.1-dev --lithe adapter.lithe \
        --prompt "a photo of a <v_sub_001> cat on the beach" --out ./outputs/samples
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch

import lithe


def attach_native(transformer, ad: lithe.LitheAdapter) -> int:
    if ad.key_format != "diffusers":
        raise ValueError("native mode needs diffusers module names; use --mode peft")
    names = [m["name"].removeprefix("transformer.") for m in ad.modules]
    layers = lithe.inject_lithe(transformer, names, rank=ad.rank, alpha=ad.alpha,
                                codebook=lithe.TwinCodebook(ad.block_size, ad.n_bits, ad.sigma_a, ad.sigma_b))
    for m, name in zip(ad.modules, names):
        idx = ad.indices[m["name"]]
        layers[name].load_indices(torch.from_numpy(idx["A"].astype(np.int64)),
                                  torch.from_numpy(idx["B"].astype(np.int64)))
    return len(layers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", default=os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev"))
    ap.add_argument("--lithe", help=".lithe file (omit for the base model)")
    ap.add_argument("--prompt", action="append", required=True, help="repeatable")
    ap.add_argument("--out", default="./outputs/samples")
    ap.add_argument("--mode", choices=["peft", "native"], default="peft")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(args.pretrained, torch_dtype=torch.bfloat16).to(args.device)
    if args.lithe:
        ad = lithe.load_lithe(args.lithe)
        if args.mode == "peft":
            pipe.load_lora_weights(lithe.to_lora_state_dict(ad, dtype=torch.bfloat16), adapter_name="lithe")
            print(f"[infer] loaded {len(ad.modules)} LoRA modules via load_lora_weights")
        else:
            print(f"[infer] attached {attach_native(pipe.transformer, ad)} LitheLinear modules")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(args.prompt):
        g = torch.Generator(device=args.device).manual_seed(args.seed + i)
        with torch.no_grad():
            img = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                       height=args.height, width=args.width, generator=g).images[0]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", prompt)[:60].strip("_")
        path = out / f"{i:02d}_{slug}.png"
        img.save(path)
        print(f"[infer] {path}")


if __name__ == "__main__":
    main()
