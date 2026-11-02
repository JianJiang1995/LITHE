#!/usr/bin/env python
"""Op-level SGMV head-to-head on the real FLUX.1-dev LoRA shapes (no FLUX weights needed).

Paper: Tab. "Batched SGMV kernel, one full 304-layer pass" (B = 96, seq 256):
our two-stage Triton SGMV 195.0 ms vs. the vLLM release of the Punica /
S-LoRA SGMV kernel 473.9 ms, numerically identical (worst cosine 1.0000).

For every distinct (d_in, d_out, r) of the adapter layout (read from a
``.lithe`` header; the Kohya-304 layout has 7 distinct shapes), both kernels
run on the same random bf16 inputs with B heterogeneous adapters; per-shape
times are weighted by how often the shape occurs to give one full LoRA-delta
pass. The bmm reference is timed as well. The vLLM kernels are vendored in
``serving/third_party/vllm_sgmv``.

    python serving/bench_sgmv_kernel.py --adapter-dir ./adapters/dreambooth30 --batch 96 --seq 256
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import torch

from _common import Log, base_parser, env_info, list_adapters, save_json, setup_device, sync_time

from lithe import read_header
from lithe.kernels import per_sample_lora_delta, per_sample_lora_delta_reference


def main():
    ap = base_parser(__doc__, flux=False)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-vllm", action="store_true", help="skip the vendored vLLM kernels")
    ap.add_argument("--vllm-no-libentry", action="store_true",
                    help="run the vLLM kernels without the libentry launch cache")
    args = ap.parse_args()
    dev = setup_device(args.gpu)
    log = Log()

    header = read_header(list_adapters(args.adapter_dir, 1)[0])
    shapes = Counter((m["shape_a"][1], m["shape_b"][0], m["shape_a"][0]) for m in header["modules"])
    log(f"{sum(shapes.values())} layers, {len(shapes)} distinct (d_in, d_out, r) shapes, "
        f"B={args.batch}, seq={args.seq}")

    vllm = None
    if not args.no_vllm:
        if args.vllm_no_libentry:
            os.environ["LITHE_VLLM_NO_LIBENTRY"] = "1"
        sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
        import vllm_sgmv as vllm

    B, S = args.batch, args.seq

    def timed(fn):
        for _ in range(args.warmup):
            fn()
        t0 = sync_time()
        for _ in range(args.iters):
            fn()
        return (sync_time() - t0) / args.iters * 1e3

    rows, tot = [], Counter()
    for (d_in, d_out, r), count in sorted(shapes.items(), key=lambda kv: -kv[1]):
        torch.manual_seed(0)
        a = (torch.randn(B, r, d_in, device=dev) * 0.02).bfloat16()
        b = (torch.randn(B, d_out, r, device=dev) * 0.02).bfloat16()
        x = (torch.randn(B, S, d_in, device=dev) * 0.5).bfloat16()
        s = torch.ones(B, device=dev, dtype=torch.bfloat16)
        row = {"d_in": d_in, "d_out": d_out, "rank": r, "count": count}
        ours = per_sample_lora_delta(x, a, b, s).float()
        row["ms_ours"] = timed(lambda: per_sample_lora_delta(x, a, b, s))
        row["ms_bmm"] = timed(lambda: per_sample_lora_delta_reference(x, a, b, s))
        ref = per_sample_lora_delta_reference(x, a, b, s).float()
        row["cos_vs_bmm"] = torch.nn.functional.cosine_similarity(ours.flatten(), ref.flatten(), dim=0).item()
        if vllm is not None:
            mid = torch.zeros(B * S, r, device=dev, dtype=torch.float32)
            out = torch.zeros(B * S, d_out, device=dev, dtype=torch.bfloat16)
            start = torch.arange(B, device=dev, dtype=torch.long) * S
            seqlen = torch.full((B,), S, device=dev, dtype=torch.long)
            lidx = torch.arange(B, device=dev, dtype=torch.long)
            xf = x.reshape(B * S, d_in).contiguous()

            def run_vllm():
                mid.zero_()
                vllm.sgmv_shrink(xf, a, mid, start, seqlen, lidx, B, S, B * S, 1.0)
                out.zero_()
                vllm.sgmv_expand(mid, b, out, start, seqlen, lidx, B, S, B * S, add_inputs=False)
                return out.view(B, S, d_out)

            theirs = run_vllm().float()
            diff = (ours - theirs).abs()
            row["cos_vs_vllm"] = torch.nn.functional.cosine_similarity(ours.flatten(), theirs.flatten(), dim=0).item()
            row["rel_mean_err_vs_vllm"] = (diff.mean() / ours.abs().mean().clamp_min(1e-6)).item()
            row["ms_vllm"] = timed(run_vllm)
            del mid, out, xf
        for k in ("ms_ours", "ms_bmm", "ms_vllm"):
            if k in row:
                tot[k] += row[k] * count
        rows.append(row)
        log("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        torch.cuda.empty_cache()

    summary = {f"total_{k}_per_pass": v for k, v in tot.items()}
    log("full pass: " + ", ".join(f"{k} = {v:.1f} ms" for k, v in summary.items()))
    if vllm is not None:
        summary["worst_cos_vs_vllm"] = min(r["cos_vs_vllm"] for r in rows)
        summary["worst_rel_mean_err_vs_vllm"] = max(r["rel_mean_err_vs_vllm"] for r in rows)
        log(f"numerics vs vLLM: worst cosine {summary['worst_cos_vs_vllm']:.6f}, "
            f"worst relative mean error {summary['worst_rel_mean_err_vs_vllm']:.2e}")
    save_json(args.out, {"env": env_info(), "batch": B, "seq": S, "iters": args.iters, **summary,
                         "per_shape": rows, "log": log.lines})


if __name__ == "__main__":
    main()
