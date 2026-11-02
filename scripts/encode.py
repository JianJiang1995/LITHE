#!/usr/bin/env python
"""Encode LoRA factors into a .lithe index file.

Two sources are supported:

  --lora FILE.safetensors   a dense PEFT/diffusers LoRA; its A/B factors are
                            twin-RVQ encoded post hoc (post-training quantisation;
                            lossier than native STE training)
  --checkpoint FILE.pt      an index checkpoint written by the original research
                            trainers (diffusers-fork ``discrete_adapter.pt`` or a Kohya
                            ``*.lithe_discrete.pt`` sidecar); indices are copied as is

    python scripts/encode.py --lora my_lora.safetensors --out my_lora.lithe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lithe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lora", help="dense LoRA safetensors (PEFT / diffusers keys)")
    src.add_argument("--checkpoint", help="trusted training checkpoint (pickle)")
    ap.add_argument("--out", required=True, help="output .lithe file")
    ap.add_argument("--base_model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                    help="extra header metadata, repeatable (e.g. trigger=<v_sub_001>)")
    args = ap.parse_args()

    if args.lora:
        from safetensors.torch import load_file
        ad = lithe.from_dense_lora(load_file(args.lora), base_model=args.base_model)
        src_bytes = Path(args.lora).stat().st_size
    else:
        ad = lithe.from_training_checkpoint(args.checkpoint)
        ad.base_model = args.base_model
        src_bytes = Path(args.checkpoint).stat().st_size
    for kv in args.meta:
        key, sep, value = kv.partition("=")
        if not sep:
            raise SystemExit(f"--meta expects KEY=VALUE, got {kv!r}")
        ad.metadata[key] = value
    size = lithe.save_lithe(args.out, ad)
    raw = 2 * ad.num_indices
    print(f"modules        {len(ad.modules)} ({ad.metadata.get('module_set', ad.key_format)}), rank {ad.rank}, alpha {ad.alpha:g}")
    print(f"indices        {ad.num_indices}  (raw uint16 {raw / 1e6:.2f} MB, 9-bit packed {ad.num_indices * 9 / 8e6:.2f} MB)")
    print(f"written        {args.out}  {size / 1e6:.3f} MB  (source file {src_bytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
