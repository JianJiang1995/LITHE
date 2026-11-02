#!/usr/bin/env python
"""Whole-adapter decode latency: fused bit-packed Triton kernel vs. per-module PyTorch decode.

Paper: App. "N_max co-resident inventory stress test and decode microbench"
(fused decode of all 304 Kohya linears, 4.56 ms warm mean, p99 4.59 ms, bit-exact
against the int16 path) and App. "Engineered cold-swap path" (~10 ms disk-to-GPU
from a raw int16 blob). No FLUX weights are needed.

Measured paths, all CUDA-synchronised:
  fused_packed    packed words already on GPU -> fused decode + per-module views
  fused_int16     same kernel family on unpacked int16 indices
  torch_loop      per-module ``lithe.quant.decode`` loop (fp32 -> bf16), indices on GPU
  cold_raw_int16  memory-mapped raw int16 blob on disk -> H2D -> fused int16 decode
  cold_lithe      .lithe file (LZMA) -> parse -> pack -> H2D -> fused packed decode

    python serving/bench_decode.py --adapter-dir ./adapters/dreambooth30 --gpu 0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from _common import (Log, PackedBank, base_parser, env_info, list_adapters, load_adapters, save_json,
                     setup_device, stats_ms, sync_time)

import lithe
from lithe import quant
from lithe.kernels import decode_packed, pack_adapter, split_decoded
from lithe.kernels.packed import decode_int16, flat_indices


def main():
    ap = base_parser(__doc__, flux=False)
    ap.add_argument("--n-adapters", type=int, default=5)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--cold-trials", type=int, default=30)
    ap.add_argument("--scratch", default="./outputs/bench_decode_scratch", help="where raw int16 blobs are written")
    args = ap.parse_args()
    dev = setup_device(args.gpu)
    log = Log()

    paths = list_adapters(args.adapter_dir, args.n_adapters)
    ads = load_adapters(paths)
    bank, words = PackedBank.from_adapters(ads, dev)
    for w in words:
        bank.add(w)
    lay = bank.layout
    log(f"{len(ads)} adapters, {len(lay.names)} modules, {lay.total_blocks} blocks, "
        f"{lay.num_indices} indices, packed {bank.slot_bytes / 1e6:.3f} MB/adapter")

    # correctness: fused packed == fused int16 == per-module PyTorch decode (bf16)
    idx16 = [torch.from_numpy(flat_indices(ad, lay).astype(np.int16)).view(-1, lay.num_levels).to(dev) for ad in ads]
    out_p = decode_packed(bank.slots[0], lay, bank.pool_a, bank.pool_b)
    out_i = decode_int16(idx16[0], lay, bank.pool_a, bank.pool_b)
    ref = lithe.decode_factors(ads[0], device=dev)
    got = split_decoded(out_p, lay)
    max_ref = max(max((got[n][0].float() - a.to(torch.bfloat16).float()).abs().max().item(),
                      (got[n][1].float() - b.to(torch.bfloat16).float()).abs().max().item())
                  for n, (a, b) in ref.items())
    max_int16 = (out_p.float() - out_i.float()).abs().max().item()
    log(f"max|packed - int16| = {max_int16:.3g}, max|packed - torch decode (bf16)| = {max_ref:.3g}")

    def timed(fn, n, warmup=3):
        for i in range(warmup):
            fn(i)
        ts = []
        for i in range(n):
            t0 = sync_time()
            fn(i)
            ts.append((sync_time() - t0) * 1000)
        return stats_ms(ts)

    def fused_packed(i):
        split_decoded(decode_packed(bank.slots[i % len(ads)], lay, bank.pool_a, bank.pool_b), lay)

    def fused_int16(i):
        split_decoded(decode_int16(idx16[i % len(ads)], lay, bank.pool_a, bank.pool_b), lay)

    gpu_idx = [{m["name"]: (torch.from_numpy(ad.indices[m["name"]]["A"].astype(np.int64)).to(dev),
                            torch.from_numpy(ad.indices[m["name"]]["B"].astype(np.int64)).to(dev))
                for m in ad.modules} for ad in ads]

    def torch_loop(i):
        k = i % len(ads)
        for m in ads[k].modules:
            ia, ib = gpu_idx[k][m["name"]]
            quant.decode(ia, m["shape_a"], bank.pool_a).to(torch.bfloat16)
            quant.decode(ib, m["shape_b"], bank.pool_b).to(torch.bfloat16)

    res = {"env": env_info(), "n_adapters": len(ads), "n_modules": len(lay.names),
           "num_indices": lay.num_indices, "packed_mb": bank.slot_bytes / 1e6,
           "int16_mb": lay.num_indices * 2 / 1e6,
           "correctness": {"max_abs_packed_vs_int16": max_int16, "max_abs_packed_vs_torch_decode": max_ref}}
    res["fused_packed_ms"] = timed(fused_packed, args.trials)
    log(f"fused packed decode: {res['fused_packed_ms']['mean']:.3f} ms mean, p99 {res['fused_packed_ms']['p99']:.3f}")
    res["fused_int16_ms"] = timed(fused_int16, args.trials)
    log(f"fused int16 decode:  {res['fused_int16_ms']['mean']:.3f} ms mean")
    res["torch_loop_ms"] = timed(torch_loop, min(args.trials, 30))
    log(f"per-module torch decode: {res['torch_loop_ms']['mean']:.2f} ms mean")

    # cold paths
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    blobs = []
    for p, ad in zip(paths, ads):
        blob = scratch / f"{p.stem}.int16.bin"
        flat_indices(ad, lay).astype(np.int16).tofile(blob)
        blobs.append(blob)

    def cold_raw(i):
        arr = np.memmap(blobs[i % len(blobs)], dtype=np.int16, mode="c")
        t = torch.from_numpy(np.ascontiguousarray(arr)).pin_memory().to(dev, non_blocking=True)
        split_decoded(decode_int16(t.view(-1, lay.num_levels), lay, bank.pool_a, bank.pool_b), lay)

    def cold_lithe(i):
        ad = lithe.load_lithe(paths[i % len(paths)])
        w, _ = pack_adapter(ad, lay)
        split_decoded(decode_packed(w.to(dev), lay, bank.pool_a, bank.pool_b), lay)

    res["cold_raw_int16_ms"] = timed(cold_raw, args.cold_trials)
    log(f"cold raw int16 blob -> GPU -> decode: {res['cold_raw_int16_ms']['mean']:.2f} ms mean "
        "(page cache warm after the first read)")
    res["cold_lithe_ms"] = timed(cold_lithe, args.cold_trials, warmup=1)
    log(f"cold .lithe (LZMA) -> pack -> GPU -> decode: {res['cold_lithe_ms']['mean']:.1f} ms mean")
    res["log"] = log.lines
    save_json(args.out, res)


if __name__ == "__main__":
    main()
