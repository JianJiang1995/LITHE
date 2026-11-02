#!/usr/bin/env python
"""Serial LoRA serving baselines: one render per adapter, with every swap strategy we compared.

Paper: "Serial fuse" column of Tab. "Same-batch wall-time" (App. "Serial-fuse
re-measurement") and Tab. "Serving-path comparison" (App. "Modern paging
comparison"; B = 96 distinct adapters, 256^2 / 4 steps).

Tiers (``--tier``), each serving ``--n`` requests, one adapter per request:
  serial-disk      load_lora_weights(file) -> fuse_lora -> render -> unfuse_lora ->
                   unload_lora_weights. ``--evict-cmd`` (run before every request,
                   outside the timed window) reproduces the cold-cache protocol.
  serial-pinned    adapters converted once into a pinned host pool (untimed); per request
                   H2D page-in -> fuse (W += s B A) -> render -> unfuse
  async-prefetch   as serial-pinned, but the page-in of request i+1 runs on a copy stream
                   during the render of request i (double buffer + CUDA events)
  hotswap          diffusers PEFT hotswap: enable_lora_hotswap + torch.compile, per request
                   load_lora_weights(..., hotswap=True) -> render (unfused)
  hotswap-steelman hotswap with conversion amortised: per request an in-place copy of
                   pre-converted pinned factors into the PEFT weights -> compiled render

Adapters: ``--lora-dir`` with rank-r LoRA ``*.safetensors`` files, or
``--from-lithe`` to decode the ``.lithe`` adapters into Kohya-key LoRA files
under ``--scratch`` first (identical shapes and file sizes to a bf16 Kohya LoRA).

    python serving/bench_paging.py --pretrained $FLUX --adapter-dir ./adapters --from-lithe \
        --tier async-prefetch --n 96 --trials 3 --out paging_async.json
"""
from __future__ import annotations

import statistics
import subprocess
import time
from pathlib import Path

import torch

from _common import (Log, base_parser, env_info, list_adapters, load_adapters, load_pipe, save_json,
                     setup_device, vram_peak_gb, write_dense_lora_from_lithe)


def render_once(pipe, args, i):
    g = torch.Generator(device="cuda").manual_seed(42 + i)
    pipe(prompt=args.prompt, height=args.size, width=args.size, num_inference_steps=args.steps,
         guidance_scale=3.5, max_sequence_length=128, generator=g).images


def load_file(pipe, path, **kw):
    pipe.load_lora_weights(str(Path(path).parent), weight_name=Path(path).name, **kw)


def lora_targets(pipe):
    """{module name: (lora_A weight, lora_B weight, scaling)} of the loaded PEFT adapter."""
    out = {}
    for name, mod in pipe.transformer.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "scaling"):
            keys = list(mod.lora_A.keys())
            if keys:
                k = keys[0]
                out[name] = (mod.lora_A[k].weight, mod.lora_B[k].weight, float(mod.scaling[k]))
    return out


def harvest_pinned(pipe, files, log):
    """Convert every adapter once through diffusers, keep (A, B, scale) in pinned host memory."""
    adapters = []
    for j, p in enumerate(files):
        load_file(pipe, p)
        adapters.append({n: (a.detach().to("cpu", torch.bfloat16).contiguous().pin_memory(),
                             b.detach().to("cpu", torch.bfloat16).contiguous().pin_memory(), s)
                         for n, (a, b, s) in lora_targets(pipe).items()})
        pipe.unload_lora_weights()
        if (j + 1) % 16 == 0:
            log(f"  harvested {j + 1}/{len(files)}")
    base = {n: m for n, m in pipe.transformer.named_modules()
            if isinstance(m, torch.nn.Linear) and n in adapters[0]}
    return adapters, base


# ---------------------------------------------------------------------------
def trial_serial_disk(pipe, files, args):
    per = []
    torch.cuda.reset_peak_memory_stats()
    wall = 0.0
    for i, f in enumerate(files):
        if args.evict_cmd:
            subprocess.run(args.evict_cmd, shell=True, check=True)
        rec = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        load_file(pipe, f)
        torch.cuda.synchronize(); rec["t_load"] = time.perf_counter() - t0
        t = time.perf_counter(); pipe.fuse_lora()
        torch.cuda.synchronize(); rec["t_fuse"] = time.perf_counter() - t
        t = time.perf_counter(); render_once(pipe, args, i)
        torch.cuda.synchronize(); rec["t_render"] = time.perf_counter() - t
        t = time.perf_counter(); pipe.unfuse_lora(); pipe.unload_lora_weights()
        torch.cuda.synchronize(); rec["t_unfuse_unload"] = time.perf_counter() - t
        rec["t_total"] = time.perf_counter() - t0
        wall += rec["t_total"]        # the eviction command is outside the timed window
        per.append(rec)
    return per, wall, vram_peak_gb()


def _apply_delta(base, factors, sign):
    for name, (a, b, s) in factors.items():
        delta = (b @ a).mul_(s)
        w = base[name].weight
        w.data.add_(delta) if sign > 0 else w.data.sub_(delta)


def trial_serial_pinned(pipe, adapters, base, args):
    torch.cuda.reset_peak_memory_stats()
    per, prev = [], None
    torch.cuda.synchronize()
    wall0 = time.time()
    for i in range(len(adapters)):
        rec = {}
        torch.cuda.synchronize(); t = time.time()
        if prev is not None:
            _apply_delta(base, prev, -1); torch.cuda.synchronize()
        rec["t_unfuse"] = time.time() - t
        t = time.time()
        cur = {n: (a.to("cuda", non_blocking=True), b.to("cuda", non_blocking=True), s)
               for n, (a, b, s) in adapters[i].items()}
        torch.cuda.synchronize(); rec["t_load"] = time.time() - t
        t = time.time(); _apply_delta(base, cur, +1)
        torch.cuda.synchronize(); rec["t_fuse"] = time.time() - t
        t = time.time(); render_once(pipe, args, i)
        torch.cuda.synchronize(); rec["t_render"] = time.time() - t
        per.append(rec)
        prev = cur
    wall = time.time() - wall0
    peak = vram_peak_gb()
    _apply_delta(base, prev, -1)
    return per, wall, peak


def trial_async_prefetch(pipe, adapters, base, args):
    n = len(adapters)

    def alloc():
        return {k: (torch.empty_like(a, device="cuda"), torch.empty_like(b, device="cuda"))
                for k, (a, b, _) in adapters[0].items()}

    bufs = [alloc(), alloc()]
    copy_stream, default = torch.cuda.Stream(), torch.cuda.current_stream()
    copy_done = [torch.cuda.Event(), torch.cuda.Event()]
    unfuse_done = [torch.cuda.Event(), torch.cuda.Event()]
    unfuse_recorded = [False, False]
    cp0 = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    cp1 = [torch.cuda.Event(enable_timing=True) for _ in range(n)]

    def prefetch(i, slot):
        with torch.cuda.stream(copy_stream):
            if unfuse_recorded[slot]:
                copy_stream.wait_event(unfuse_done[slot])   # slot still read by the last unfuse
            cp0[i].record(copy_stream)
            for k, (a, b, _) in adapters[i].items():
                ba, bb = bufs[slot][k]
                ba.copy_(a, non_blocking=True)
                bb.copy_(b, non_blocking=True)
            cp1[i].record(copy_stream)
            copy_done[slot].record(copy_stream)

    def apply(slot, i, sign):
        for k, (_, _, s) in adapters[i].items():
            ba, bb = bufs[slot][k]
            delta = (bb @ ba).mul_(s)
            w = base[k].weight
            w.data.add_(delta) if sign > 0 else w.data.sub_(delta)

    ev = {k: [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
          for k in ("r0", "r1", "u0", "u1", "w0", "w1", "f0", "f1")}
    torch.cuda.reset_peak_memory_stats()
    prefetch(0, 0)
    default.wait_event(copy_done[0])
    torch.cuda.synchronize()
    wall0 = time.time()
    apply(0, 0, +1)
    for i in range(n):
        cur, nxt = i % 2, (i + 1) % 2
        if i + 1 < n:
            prefetch(i + 1, nxt)                          # overlaps the render below
        ev["r0"][i].record(); render_once(pipe, args, i); ev["r1"][i].record()
        ev["u0"][i].record(); apply(cur, i, -1); ev["u1"][i].record()
        unfuse_done[cur].record(default); unfuse_recorded[cur] = True
        if i + 1 < n:
            ev["w0"][i].record(); default.wait_event(copy_done[nxt]); ev["w1"][i].record()
            ev["f0"][i].record(); apply(nxt, i + 1, +1); ev["f1"][i].record()
    torch.cuda.synchronize()
    wall = time.time() - wall0
    per = []
    for i in range(n):
        rec = {"t_render": ev["r0"][i].elapsed_time(ev["r1"][i]) / 1e3,
               "t_unfuse": ev["u0"][i].elapsed_time(ev["u1"][i]) / 1e3}
        if i + 1 < n:
            rec["t_wait_load"] = ev["w0"][i].elapsed_time(ev["w1"][i]) / 1e3
            rec["t_fuse"] = ev["f0"][i].elapsed_time(ev["f1"][i]) / 1e3
            rec["t_pagein_hidden"] = cp0[i + 1].elapsed_time(cp1[i + 1]) / 1e3
        per.append(rec)
    return per, wall, vram_peak_gb()


def _recompiles():
    try:
        import torch._dynamo as dyn
        return dyn.utils.counters["stats"].get("unique_graphs", 0)
    except Exception:
        return 0


def setup_hotswap(pipe, files, args, log, steelman: bool):
    pipe.enable_lora_hotswap(target_rank=args.rank)
    load_file(pipe, files[0], adapter_name="default_0")
    pinned = None
    if steelman:   # convert every adapter once, keep the PEFT-ready factors pinned on the host
        pinned = []
        for j, p in enumerate(files):
            if j > 0:
                load_file(pipe, p, adapter_name="default_0", hotswap=True)
            pinned.append({n: (a.detach().to("cpu", torch.bfloat16).contiguous().pin_memory(),
                               b.detach().to("cpu", torch.bfloat16).contiguous().pin_memory())
                           for n, (a, b, _) in lora_targets(pipe).items()})
    targets = lora_targets(pipe)
    if args.compile != "none":
        pipe.transformer = torch.compile(pipe.transformer, mode=args.compile, fullgraph=False)
    try:
        import torch._dynamo as dyn
        dyn.config.cache_size_limit = 64
    except Exception:
        pass
    t0 = time.time()
    render_once(pipe, args, 0)
    render_once(pipe, args, 0)
    torch.cuda.synchronize()
    log(f"hotswap warm-up (compile={args.compile}) {time.time() - t0:.1f} s")
    return targets, pinned


def trial_hotswap(pipe, files, args, targets, pinned):
    torch.cuda.reset_peak_memory_stats()
    per = []
    rc0 = _recompiles()
    torch.cuda.synchronize()
    wall0 = time.time()
    for i, f in enumerate(files):
        rec = {}
        torch.cuda.synchronize(); t = time.time()
        if pinned is None:
            load_file(pipe, f, adapter_name="default_0", hotswap=True)
        else:
            for n, (aw, bw, _) in targets.items():
                pa, pb = pinned[i][n]
                aw.data.copy_(pa, non_blocking=True)
                bw.data.copy_(pb, non_blocking=True)
        torch.cuda.synchronize(); rec["t_load"] = time.time() - t
        t = time.time(); render_once(pipe, args, i)
        torch.cuda.synchronize(); rec["t_render"] = time.time() - t
        per.append(rec)
    wall = time.time() - wall0
    return per, wall, vram_peak_gb(), _recompiles() - rc0


def main():
    ap = base_parser(__doc__, adapters_required=False)
    ap.add_argument("--tier", required=True,
                    choices=["serial-disk", "serial-pinned", "async-prefetch", "hotswap", "hotswap-steelman"])
    ap.add_argument("--lora-dir", default=None, help="folder of LoRA *.safetensors (one adapter per file)")
    ap.add_argument("--from-lithe", action="store_true", help="decode the .lithe adapters into LoRA files")
    ap.add_argument("--scratch", default="./outputs/bench_paging_loras")
    ap.add_argument("--n", type=int, default=96, help="requests = distinct adapters (cycled if fewer files)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=6, help="requests excluded from the per-swap means")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt", default="a stylish portrait, full body")
    ap.add_argument("--rank", type=int, default=16, help="hotswap target rank")
    ap.add_argument("--compile", default="default", choices=["none", "default", "reduce-overhead"],
                    help="hotswap tiers only")
    ap.add_argument("--evict-cmd", default=None,
                    help="serial-disk: shell command run before every request, outside the timer "
                         "(e.g. 'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches')")
    args = ap.parse_args()
    setup_device(args.gpu)
    log = Log()

    if args.lora_dir:
        files = sorted(Path(args.lora_dir).rglob("*.safetensors"))
    elif args.from_lithe and args.adapter_dir:
        paths = list_adapters(args.adapter_dir, args.n)
        files = []
        for p, ad in zip(paths, load_adapters(paths)):
            f = Path(args.scratch) / f"{p.stem}.safetensors"
            if not f.exists():
                write_dense_lora_from_lithe(ad, f)
            files.append(f)
    else:
        raise SystemExit("need --lora-dir, or --from-lithe with --adapter-dir")
    files = [files[i % len(files)] for i in range(args.n)]
    log(f"tier={args.tier}: {args.n} requests over {len(set(files))} distinct LoRA files, "
        f"{args.size}^2 / {args.steps} steps")

    pipe = load_pipe(args.pretrained, "cuda")
    extra = {}
    if args.tier in ("serial-pinned", "async-prefetch"):
        adapters, base = harvest_pinned(pipe, files, log)
        extra["pinned_mb_per_adapter"] = sum(a.numel() * 2 + b.numel() * 2 for a, b, _ in adapters[0].values()) / 1e6
        fn = lambda: (trial_serial_pinned if args.tier == "serial-pinned" else trial_async_prefetch)(  # noqa: E731
            pipe, adapters, base, args)
    elif args.tier == "serial-disk":
        load_file(pipe, files[0]); pipe.fuse_lora(); render_once(pipe, args, 0)
        pipe.unfuse_lora(); pipe.unload_lora_weights()
        fn = lambda: trial_serial_disk(pipe, files, args)  # noqa: E731
    else:
        targets, pinned = setup_hotswap(pipe, files, args, log, steelman=args.tier == "hotswap-steelman")
        fn = lambda: trial_hotswap(pipe, files, args, targets, pinned)  # noqa: E731

    trials = []
    for t in range(args.trials):
        out = fn()
        per, wall, peak = out[:3]
        body = per[args.warmup:] or per
        means = {k: statistics.mean(r[k] for r in body if k in r) for k in body[0]}
        row = {"trial": t + 1, "wall_s": wall, "peak_alloc_gb": peak, "mean_per_request_s": means}
        if len(out) > 3:
            row["recompiles"] = out[3]
        trials.append(row)
        log(f"trial {t + 1}: wall {wall:.2f} s, peak {peak:.1f} GB, per request: "
            + " ".join(f"{k}={v:.3f}" for k, v in means.items()))
    walls = [r["wall_s"] for r in trials]
    res = {"env": env_info(), "tier": args.tier, "n": args.n, "size": args.size, "steps": args.steps,
           "wall_mean_s": statistics.mean(walls), "wall_std_s": statistics.stdev(walls) if len(walls) > 1 else 0.0,
           "trials": trials, **extra, "log": log.lines}
    log(f"{args.tier}: {res['wall_mean_s']:.2f} +- {res['wall_std_s']:.2f} s")
    save_json(args.out, res)


if __name__ == "__main__":
    main()
