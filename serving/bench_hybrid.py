#!/usr/bin/env python
"""Hybrid N_res x B_serv sweep: large packed inventory vs. same-batch heterogeneity.

Paper: Tab. "Hybrid N_res x B_serv operating points" (main text, P2-P6) and
Tab. "Hybrid ... operating-point sweep" (appendix, P1-P5 at 512^2 / 16 steps).

For every point (N, B): build a packed bank of N slots (real adapters first,
then clones of slot 0), decode B adapters from it with the fused kernel
(cycling over the real ones when B exceeds them), install them as per-sample
factors, and render one batch of B different prompts. The fused decode and the
forward path are warmed up once before the sweep (as in the paper) so the
first point does not absorb Triton JIT / autotune time.

    python serving/bench_hybrid.py --pretrained $FLUX --adapter-dir ./adapters/dreambooth30 \
        --points 11000x1 8000x4 4000x8 1000x14 4x14 --size 512 --steps 16 --out hybrid.json
"""
from __future__ import annotations

from pathlib import Path

import torch

from _common import (Log, PackedBank, adapter_label, adapter_prompt, base_parser, env_info,
                     free_cuda, install_per_sample, list_adapters, load_adapters, load_pipe, render,
                     save_json, setup_device, sync_time, vram_alloc_gb, vram_peak_gb, vram_reserved_gb,
                     wrap_decoded)

from lithe import kohya


def parse_point(s: str):
    n, _, b = s.lower().partition("x")
    return int(n), int(b)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--points", nargs="+", default=["11000x1", "8000x4", "4000x8", "1000x14", "4x14"],
                    help="N_res x B_serv operating points")
    ap.add_argument("--n-real", type=int, default=8, help="distinct real adapters (the paper used 8)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--kernel", choices=["loop", "sgmv"], default="loop",
                    help="per-sample delta of the wrapper (the paper run used the loop wrapper)")
    ap.add_argument("--save-images", default=None)
    args = ap.parse_args()
    dev = setup_device(args.gpu)
    log = Log()

    pipe = load_pipe(args.pretrained, dev)
    base_gb = vram_alloc_gb()
    log(f"FLUX.1-dev base: {base_gb:.2f} GB allocated")
    paths = list_adapters(args.adapter_dir, args.n_real)
    ads = load_adapters(paths)
    bank, words = PackedBank.from_adapters(ads, dev)
    key_format = ads[0].key_format
    wrapped = wrap_decoded(pipe.transformer, kohya.diffusers_paths(ads[0]), args.kernel)
    log(f"wrapped {len(wrapped)} linears; {len(ads)} real adapters")

    # one-time warm-up of the fused decode and the forward path
    warm = PackedBank(bank.layout, dev)
    warm.add(words[0])
    for _ in range(3):
        install_per_sample(wrapped, [warm.decode_diffusers(0, key_format)])
    render(pipe, ["warmup"], 256, 4)
    del warm
    free_cuda()

    results = []
    for spec in args.points:
        n, b = parse_point(spec)
        name = f"N{n}_B{b}"
        bank.slots.clear()
        free_cuda()
        for w in words:
            bank.add(w)
        try:
            bank.fill_to(n)
            torch.cuda.synchronize()
            resident_gb, resident_res = vram_alloc_gb(), vram_reserved_gb()
            log(f"[{name}] {len(bank.slots)} slots resident: {resident_gb:.2f} GB")
            sel = [i % len(ads) for i in range(b)]
            torch.cuda.reset_peak_memory_stats()
            t0 = sync_time()
            decoded = [bank.decode_diffusers(i, key_format) for i in sel]
            install_per_sample(wrapped, decoded)
            dec_ms = (sync_time() - t0) * 1e3
            images, wall = render(pipe, [adapter_prompt(ads[i]) for i in sel], args.size, args.steps)
            row = {"point": name, "n_res": len(bank.slots), "b_serv": b, "distinct": len(set(sel)),
                   "resident_alloc_gb": resident_gb, "resident_reserved_gb": resident_res,
                   "peak_alloc_gb": vram_peak_gb(), "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                   "decode_install_ms": dec_ms, "forward_s": wall, "ms_per_sample": wall * 1e3 / b,
                   "config": f"{args.size}x{args.size}/{args.steps}", "adapters": [adapter_label(paths[i]) for i in sel]}
            log(f"[{name}] decode+install {dec_ms:.1f} ms, forward {wall:.2f} s "
                f"({row['ms_per_sample']:.0f} ms/sample), peak {row['peak_alloc_gb']:.2f} GB")
            if args.save_images:
                d = Path(args.save_images) / name
                d.mkdir(parents=True, exist_ok=True)
                for k, (img, i) in enumerate(zip(images, sel)):
                    img.save(d / f"b{k:02d}_{adapter_label(paths[i])}.jpg", quality=90)
            del decoded, images
        except torch.cuda.OutOfMemoryError as e:
            row = {"point": name, "n_res": n, "b_serv": b, "oom": True, "error": str(e)[:200]}
            log(f"[{name}] OOM")
        for mod in wrapped.values():
            mod.clear()
        results.append(row)

    save_json(args.out, {"env": env_info(), "base_alloc_gb": base_gb, "wrapped": len(wrapped),
                         "points": results, "log": log.lines})


if __name__ == "__main__":
    main()
