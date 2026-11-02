#!/usr/bin/env python
"""Bit-packed index residency on one GPU next to the bf16 FLUX.1-dev pipeline.

Paper: Tab. "Allocator accounting for the verified 10,000-slot index-resident
pool" (33.76 GB base, 75.70 GB after 10,000 slots with buffers and wrappers,
4.19 MB per slot), App. "N_max co-resident inventory stress test" (OOM probe),
and Tab. "End-to-end serving from a 10,000-adapter resident pool" (hot-swap
450 ms cold / ~9.5 ms warm, ~14 s per 1024^2 / 28-step render).

Steps:
  1. load the FLUX.1-dev pipeline (bf16) and record the base allocation;
  2. put the real adapters into the first slots of a packed bank, fill it to
     ``--n-slots`` by cloning slot 0 (same bytes, distinct allocations);
  3. optional ``--probe-oom``: keep adding slots in tiers until OOM, then
     binary-search the boundary (allocator-sensitive estimate);
  4. optional ``--renders K``: wrap the 494 diffusers linears with the
     per-sample wrapper, then K times hot-swap to a real resident adapter
     (fused decode + Kohya->diffusers split + install) and render one image.

    python serving/bench_residency.py --pretrained $FLUX --adapter-dir ./adapters/dreambooth30 \
        --n-slots 10000 --renders 3 --size 1024 --steps 28 --out residency.json
"""
from __future__ import annotations

from pathlib import Path

import torch

from _common import (Log, PackedBank, adapter_label, adapter_prompt, base_parser, env_info,
                     free_cuda, install_per_sample, list_adapters, load_adapters, load_pipe, render,
                     save_json, setup_device, sync_time, vram_alloc_gb, vram_peak_gb, vram_reserved_gb,
                     wrap_decoded)

from lithe import kohya


def probe_oom(bank: PackedBank, base_gb: float, tiers, log, gap: int = 200):
    """Grow the bank through ``tiers`` until OOM, then bisect to within ``gap`` slots."""
    growth, last_ok, first_oom = [], len(bank.slots), None
    for target in tiers:
        try:
            bank.fill_to(target)
            torch.cuda.synchronize()
            cur = vram_alloc_gb()
            growth.append({"n": target, "alloc_gb": cur, "mb_per_slot": (cur - base_gb) / target * 1e3})
            log(f"  N={target}: alloc {cur:.2f} GB")
            last_ok = target
        except torch.cuda.OutOfMemoryError:
            first_oom = target
            log(f"  OOM at N={target}")
            break
    if first_oom is not None:
        lo, hi = last_ok, first_oom
        while hi - lo > gap:
            mid = (lo + hi) // 2
            try:
                bank.shrink_to(min(len(bank.slots), mid))
                free_cuda()
                bank.fill_to(mid)
                torch.cuda.synchronize()
                lo = mid
                log(f"  bisect N={mid}: fits ({vram_alloc_gb():.2f} GB)")
            except torch.cuda.OutOfMemoryError:
                hi = mid
                bank.shrink_to(lo)
                free_cuda()
                log(f"  bisect N={mid}: OOM")
    return {"tiers": growth, "last_fit": last_ok, "first_oom": first_oom}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n-slots", type=int, default=10000)
    ap.add_argument("--n-real", type=int, default=3, help="real adapters placed in the first slots")
    ap.add_argument("--probe-oom", action="store_true", help="search the OOM boundary instead of rendering")
    ap.add_argument("--probe-tiers", type=int, nargs="+", default=[1000, 3000, 6000, 9000, 12000, 15000])
    ap.add_argument("--renders", type=int, default=3, help="hot-swap + render cycles (0 = residency only)")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--kernel", choices=["loop", "sgmv"], default="loop",
                    help="per-sample delta of the wrapper (the paper run used the loop wrapper)")
    ap.add_argument("--save-images", default=None, help="folder for the rendered images")
    args = ap.parse_args()
    dev = setup_device(args.gpu)
    log = Log()

    pipe = load_pipe(args.pretrained, dev)
    base_gb = vram_alloc_gb()
    log(f"FLUX.1-dev base: {base_gb:.2f} GB allocated")

    paths = list_adapters(args.adapter_dir, args.n_real)
    ads = load_adapters(paths)
    bank, words = PackedBank.from_adapters(ads, dev)
    for w in words:
        bank.add(w)
    log(f"packed slot: {bank.slot_bytes / 1e6:.3f} MB logical ({bank.layout.num_indices} indices)")

    res = {"env": env_info(), "base_alloc_gb": base_gb, "slot_logical_mb": bank.slot_bytes / 1e6,
           "n_real": len(ads), "adapters": [adapter_label(p) for p in paths]}

    if args.probe_oom:
        res["probe"] = probe_oom(bank, base_gb, [t for t in args.probe_tiers if t > len(bank.slots)], log)
        res["log"] = log.lines
        save_json(args.out, res)
        return

    t0 = sync_time()
    bank.fill_to(args.n_slots)
    torch.cuda.synchronize()
    res["fill_s"] = sync_time() - t0
    res["n_slots"] = len(bank.slots)
    res["resident_alloc_gb"] = vram_alloc_gb()
    res["resident_reserved_gb"] = vram_reserved_gb()
    res["mb_per_slot_incl_overhead"] = (res["resident_alloc_gb"] - base_gb) / len(bank.slots) * 1e3
    log(f"N={len(bank.slots)} slots resident: {res['resident_alloc_gb']:.2f} GB allocated "
        f"({res['mb_per_slot_incl_overhead']:.3f} MB/slot over base)")

    if args.renders > 0:
        paths_d = kohya.diffusers_paths(ads[0])
        wrapped = wrap_decoded(pipe.transformer, paths_d, args.kernel)
        log(f"wrapped {len(wrapped)} linears ({args.kernel} wrapper)")
        torch.cuda.reset_peak_memory_stats()
        runs = []
        for k in range(args.renders):
            slot = k % len(ads)
            t0 = sync_time()
            dec = bank.decode_diffusers(slot, ads[slot].key_format)
            install_per_sample(wrapped, [dec])
            swap_ms = (sync_time() - t0) * 1e3
            images, wall = render(pipe, [adapter_prompt(ads[slot])], args.size, args.steps)
            runs.append({"run": k, "adapter": adapter_label(paths[slot]), "slot": slot,
                         "hot_swap_ms": swap_ms, "render_s": wall})
            log(f"  run {k}: {adapter_label(paths[slot])} hot-swap {swap_ms:.2f} ms, "
                f"render {args.size}^2/{args.steps} {wall:.2f} s")
            if args.save_images:
                Path(args.save_images).mkdir(parents=True, exist_ok=True)
                images[0].save(Path(args.save_images) / f"run{k:02d}_{adapter_label(paths[slot])}.jpg", quality=92)
        res["renders"] = runs
        res["peak_alloc_gb"] = vram_peak_gb()
        res["final_alloc_gb"] = vram_alloc_gb()
        res["final_reserved_gb"] = vram_reserved_gb()
        log(f"peak allocated during serving: {res['peak_alloc_gb']:.2f} GB")
    res["log"] = log.lines
    save_json(args.out, res)


if __name__ == "__main__":
    main()
