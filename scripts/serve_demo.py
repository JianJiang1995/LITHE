#!/usr/bin/env python
"""Heterogeneous multi-adapter batch on FLUX.1-dev: one forward, several LITHE adapters.

Every wrapped linear keeps only the index streams of all loaded adapters and
decodes the active ones per forward (``lithe.serve``). Each prompt gets a route:
an adapter id, a weighted mixture, or ``none`` for the base model.

    python scripts/serve_demo.py --pretrained ./checkpoints/FLUX.1-dev \
        --lithe cat.lithe --lithe style.lithe \
        --request "0|a photo of a <v_sub_001> cat" \
        --request "1|a lighthouse in <v_sty_001> style" \
        --request "0:0.7,1:0.7|a <v_sub_001> cat in <v_sty_001> style" \
        --request "none|a lighthouse"
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

import lithe
from lithe import serve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", default=os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev"))
    ap.add_argument("--lithe", action="append", required=True, help="adapter file; ids follow the order given")
    ap.add_argument("--request", action="append", required=True, help="'ROUTE|PROMPT', ROUTE like 0 or 0:0.6,1:0.4 or none")
    ap.add_argument("--out", default="./outputs/serve_demo")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from diffusers import FluxPipeline

    routes, prompts = [], []
    for req in args.request:
        route, _, prompt = req.partition("|")
        routes.append(serve.parse_route(route))
        prompts.append(prompt)
    pipe = FluxPipeline.from_pretrained(args.pretrained, torch_dtype=torch.bfloat16).to(args.device)
    adapters = [lithe.load_lithe(p) for p in args.lithe]
    router, layers = serve.attach_adapters(pipe.transformer, adapters)
    idx_bytes = sum(l.idx_a.numel() + l.idx_b.numel() for l in layers.values()) * 2
    print(f"[serve] {len(adapters)} adapters resident as indices in {len(layers)} linears "
          f"({idx_bytes / 1e6:.1f} MB int16 on device)")
    router.set(routes)
    g = torch.Generator(device=args.device).manual_seed(args.seed)
    t0 = time.time()
    with torch.no_grad():
        images = pipe(prompt=prompts, num_inference_steps=args.steps, guidance_scale=args.guidance,
                      height=args.size, width=args.size, generator=g).images
    print(f"[serve] batch of {len(prompts)} heterogeneous requests in {time.time() - t0:.1f} s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (img, req) in enumerate(zip(images, args.request)):
        path = out / f"{i:02d}.png"
        img.save(path)
        print(f"[serve] {path}  <- {req}")


if __name__ == "__main__":
    main()
