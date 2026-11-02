#!/usr/bin/env python
"""Same-batch heterogeneous routing: B different adapters in one FLUX.1-dev forward.

Paper: Tab. "Same-batch wall-time" (LoRA+SGMV, LoRA+compile and LITHE columns),
Tab. "Component ablation of LITHE's default serving kernel", the B=96 point of
App. "Same-batch per-sample routing" / "Modern paging comparison" (LITHE
same-batch routing and "dense factors resident, same SGMV kernel" rows).

Representations (``--mode``):
  lithe-index    index slots resident in every wrapped linear; the decode is fused into
                 the shrink / expand tiles with one shared codebook pool
                 (``lithe.kernels.shared``; the paper's default LITHE serving kernel)
  lithe-decoded  adapters decoded once to bf16 factors, then ``--kernel sgmv`` (two-stage
                 Triton SGMV) or ``--kernel loop`` (per-sample Python loop)
  dense          dense rank-r LoRA factors (``--lora-dir`` *.safetensors, or decoded from
                 the .lithe files with ``--dense-from-lithe``) through the same wrappers

``--compile`` applies ``torch.compile`` to the transformer after wrapping. One
untimed warm-up render precedes the timed trials (it absorbs compile and Triton
autotune). Sample i uses adapter i mod n_adapters.

    # paper Tab. 2, LITHE column at 1024^2 / 28 steps, B = 14
    python serving/bench_same_batch.py --pretrained $FLUX --adapter-dir ./adapters \
        --mode lithe-index --compile default --batch 14 --size 1024 --steps 28 --trials 3
"""
from __future__ import annotations

import statistics
from pathlib import Path

import torch

from _common import (Log, adapter_label, adapter_prompt, base_parser, decoded_diffusers_factors,
                     env_info, free_cuda, install_per_sample, list_adapters, load_adapters,
                     load_dense_lora, load_pipe, render, save_json, select_paths, set_assign,
                     setup_device, sync_time, vram_alloc_gb, vram_peak_gb, wrap_decoded, wrap_index)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--mode", choices=["lithe-index", "lithe-decoded", "dense"], default="lithe-index")
    ap.add_argument("--kernel", choices=["sgmv", "loop"], default="sgmv", help="for lithe-decoded / dense")
    ap.add_argument("--lora-dir", default=None, help="dense mode: folder of LoRA *.safetensors")
    ap.add_argument("--dense-from-lithe", action="store_true", help="dense mode: decode the .lithe files instead")
    ap.add_argument("--compile", default="none", choices=["none", "default", "reduce-overhead",
                                                          "max-autotune-no-cudagraphs"])
    ap.add_argument("--module-set", choices=["all", "attn-ff"], default="all",
                    help="all: every linear of the adapter (494 for Kohya-304); attn-ff: the 342 "
                         "attention / FFN linears (module set of the paper's wall-time adapters)")
    ap.add_argument("--batch", type=int, default=14)
    ap.add_argument("--n-adapters", type=int, default=None, help="distinct adapters (default: batch)")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--same-prompt", default=None, help="use one prompt for every sample")
    ap.add_argument("--save-images", default=None)
    args = ap.parse_args()
    dev = setup_device(args.gpu)
    log = Log()
    n_ad = args.n_adapters or args.batch

    pipe = load_pipe(args.pretrained, dev)
    base_gb = vram_alloc_gb()
    paths = list_adapters(args.adapter_dir, n_ad)
    ads = load_adapters(paths)
    prompts = [args.same_prompt or adapter_prompt(ads[i % len(ads)]) for i in range(args.batch)]
    log(f"base {base_gb:.2f} GB; {len(ads)} distinct adapters for B={args.batch}; mode={args.mode}")

    t0 = sync_time()
    if args.mode == "lithe-index":
        wrapped = wrap_index(pipe.transformer, ads, dev, module_set=args.module_set)
        assign = torch.tensor([i % len(ads) for i in range(args.batch)], dtype=torch.int32, device=dev)
        set_assign(wrapped, assign)
    else:
        if args.mode == "lithe-decoded":
            decoded = [decoded_diffusers_factors(ad, dev) for ad in ads]
        elif args.lora_dir:
            files = sorted(Path(args.lora_dir).rglob("*.safetensors"))[:n_ad]
            if len(files) < n_ad:
                raise SystemExit(f"need {n_ad} LoRA files in {args.lora_dir}, found {len(files)}")
            decoded = [load_dense_lora(f, dev) for f in files]
        elif args.dense_from_lithe:
            decoded = [decoded_diffusers_factors(ad, dev) for ad in ads]
        else:
            raise SystemExit("dense mode needs --lora-dir or --dense-from-lithe")
        paths_d = select_paths(list(decoded[0].keys()), args.module_set)
        wrapped = wrap_decoded(pipe.transformer, paths_d, args.kernel)
        install_per_sample(wrapped, decoded)
    torch.cuda.synchronize()
    attach_s = sync_time() - t0
    resident_gb = vram_alloc_gb()
    log(f"attached {len(wrapped)} linears in {attach_s:.1f} s; allocated {resident_gb:.2f} GB "
        f"({(resident_gb - base_gb) * 1e3 / len(ads):.1f} MB per adapter)")

    if args.compile != "none":
        pipe.transformer = torch.compile(pipe.transformer, mode=args.compile)
    t0 = sync_time()
    render(pipe, prompts, args.size, args.steps)
    warm_s = sync_time() - t0
    log(f"warm-up render {warm_s:.1f} s (compile={args.compile})")

    walls, peaks = [], []
    for t in range(args.trials):
        free_cuda()
        images, wall = render(pipe, prompts, args.size, args.steps)
        walls.append(wall)
        peaks.append(vram_peak_gb())
        log(f"trial {t + 1}: wall {wall:.2f} s ({wall * 1e3 / args.batch:.0f} ms/sample), peak {peaks[-1]:.2f} GB")
        if args.save_images and t == 0:
            d = Path(args.save_images)
            d.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images):
                img.save(d / f"{i:03d}_{adapter_label(paths[i % len(ads)])}.jpg", quality=90)

    res = {"env": env_info(), "mode": args.mode, "module_set": args.module_set,
           "kernel": args.kernel if args.mode != "lithe-index" else "shared",
           "compile": args.compile, "batch": args.batch, "distinct_adapters": len(ads),
           "size": args.size, "steps": args.steps, "wrapped": len(wrapped),
           "base_alloc_gb": base_gb, "resident_alloc_gb": resident_gb, "attach_s": attach_s,
           "warmup_s": warm_s, "walls_s": walls, "wall_mean_s": statistics.mean(walls),
           "wall_std_s": statistics.stdev(walls) if len(walls) > 1 else 0.0, "peak_alloc_gb": max(peaks),
           "log": log.lines}
    log(f"wall {res['wall_mean_s']:.2f} +- {res['wall_std_s']:.2f} s over {len(walls)} trials")
    save_json(args.out, res)


if __name__ == "__main__":
    main()
