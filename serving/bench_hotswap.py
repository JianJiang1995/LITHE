#!/usr/bin/env python
"""Hot-swap latency on the wrapped FLUX.1-dev transformer (no rendering).

Paper: App. "N_max co-resident inventory" (pointer hot-swap of pre-decoded
factors, 1.171 ms mean / p99 1.240 ms over all wrapped linears, independent of
how many adapters are resident), Tab. "End-to-end serving from a 10,000-adapter
resident pool" (decode-on-swap from the packed bank, ~9.5 ms warm) and App.
"Engineered cold-swap path" (~10 ms disk-to-GPU from a raw int16 blob).

Swap paths, each CUDA-synchronised and averaged over ``--cycles`` swaps:
  pointer      pre-decoded bf16 factors of N adapters resident; a swap points every
               wrapped linear at another adapter (list update, no copy)
  packed       packed 9-bit slots resident; a swap = fused decode + Kohya->diffusers
               split + install
  cold-blob    raw int16 index blob on disk -> H2D -> fused decode + split + install

Only the transformer is loaded (the text encoders and VAE are not needed).

    python serving/bench_hotswap.py --pretrained $FLUX --adapter-dir ./adapters/dreambooth30 \
        --n-resident 8 24 40 --out hotswap.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from _common import (Log, PackedBank, base_parser, decoded_diffusers_factors, env_info,
                     install_per_sample, list_adapters, load_adapters, save_json, setup_device,
                     stats_ms, sync_time, vram_alloc_gb, wrap_decoded)

from lithe import kohya
from lithe.kernels import split_decoded
from lithe.kernels.packed import decode_int16, flat_indices


def time_swaps(fn, n_choices, cycles, warmup):
    for c in range(warmup):
        fn(c % n_choices)
    ts = []
    for c in range(cycles):
        t0 = sync_time()
        fn(c % n_choices)
        ts.append((sync_time() - t0) * 1e3)
    return stats_ms(ts)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n-resident", type=int, nargs="+", default=[8, 24, 40])
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--scratch", default="./outputs/bench_hotswap_scratch")
    args = ap.parse_args()
    dev = setup_device(args.gpu)
    log = Log()

    from diffusers import FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained, subfolder="transformer",
                                                         torch_dtype=torch.bfloat16).to(dev)
    log(f"transformer loaded: {vram_alloc_gb():.2f} GB")

    paths = list_adapters(args.adapter_dir, max(args.n_resident))
    ads = load_adapters(paths)
    key_format = ads[0].key_format
    wrapped = wrap_decoded(transformer, kohya.diffusers_paths(ads[0]), "loop")
    log(f"wrapped {len(wrapped)} linears, {len(ads)} adapters available")
    res = {"env": env_info(), "wrapped": len(wrapped), "pointer": []}

    # pointer swap over N pre-decoded resident adapters
    decoded = [decoded_diffusers_factors(ad, dev) for ad in ads]
    for n in args.n_resident:
        st = time_swaps(lambda k: install_per_sample(wrapped, [decoded[k]]), min(n, len(ads)), args.cycles, args.warmup)
        res["pointer"].append({"n_resident": min(n, len(ads)), **st})
        log(f"pointer swap, N={min(n, len(ads))}: {st['mean']:.3f} ms mean, p99 {st['p99']:.3f}")
    decoded = None
    torch.cuda.empty_cache()

    # decode-on-swap from the packed resident bank
    bank, words = PackedBank.from_adapters(ads, dev)
    for w in words:
        bank.add(w)
    st = time_swaps(lambda k: install_per_sample(wrapped, [bank.decode_diffusers(k, key_format)]),
                    len(ads), args.cycles, args.warmup)
    res["packed"] = st
    log(f"packed decode + split + install: {st['mean']:.2f} ms mean, p99 {st['p99']:.2f}")

    # cold swap from a raw int16 blob on disk
    lay = bank.layout
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    blobs = []
    for p, ad in zip(paths, ads):
        blob = scratch / f"{p.stem}.int16.bin"
        flat_indices(ad, lay).astype(np.int16).tofile(blob)
        blobs.append(blob)

    def cold(k):
        arr = np.memmap(blobs[k], dtype=np.int16, mode="c")
        idx = torch.from_numpy(np.ascontiguousarray(arr)).pin_memory().to(dev, non_blocking=True)
        fac = split_decoded(decode_int16(idx.view(-1, lay.num_levels), lay, bank.pool_a, bank.pool_b), lay)
        if key_format == "kohya":
            fac = kohya.split_factors(fac)
        install_per_sample(wrapped, [{k2: (a, b, lay.scaling) for k2, (a, b) in fac.items()}])

    st = time_swaps(cold, len(blobs), min(args.cycles, 50), 3)
    res["cold_blob"] = st
    res["blob_mb"] = lay.num_indices * 2 / 1e6
    log(f"cold raw int16 blob ({res['blob_mb']:.2f} MB) -> GPU -> decode + install: {st['mean']:.2f} ms mean "
        "(OS page cache is warm after the first read of each file)")
    res["log"] = log.lines
    save_json(args.out, res)


if __name__ == "__main__":
    main()
