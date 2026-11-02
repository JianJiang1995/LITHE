#!/usr/bin/env python
"""End-to-end LITHE pipeline on a toy model, CPU only, no FLUX weights (about 10 s).

Steps: inject LitheLinear into a two-layer toy network -> train with STE to fit a
low-rank teacher update -> write a .lithe index file -> reload it -> decode to a
dense LoRA state dict -> check that the dense LoRA reproduces the LITHE forward.

    python scripts/cpu_smoke.py --steps 300 --out ./outputs/cpu_smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

import lithe


class Toy(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def dense_forward(model: Toy, lora: dict, x: torch.Tensor) -> torch.Tensor:
    """Forward of the frozen base plus a dense LoRA given as PEFT keys (alpha folded into B)."""
    def lin(base, name, h):
        a = lora[f"transformer.{name}.lora_A.weight"].float()
        b = lora[f"transformer.{name}.lora_B.weight"].float()
        return base(h) + h @ a.T @ b.T
    h = torch.tanh(lin(model.fc1.base, "fc1", x))
    return lin(model.fc2.base, "fc2", h)


def run(steps: int = 300, rank: int = 8, d: int = 128, seed: int = 0, out: str = "./outputs/cpu_smoke",
        lr: float = 2e-3, verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    model = Toy(d)
    teacher = Toy(d)
    teacher.load_state_dict(model.state_dict())
    book = lithe.TwinCodebook()
    with torch.no_grad():  # teacher = base + a low-rank update built from random Hadamard codewords
        for lin_ in (teacher.fc1, teacher.fc2):
            shape_a, shape_b = (rank, lin_.in_features), (lin_.out_features, rank)
            idx_a = torch.randint(0, lithe.K, (lithe.NUM_LEVELS, -(-shape_a[0] * shape_a[1] // 64)))
            idx_b = torch.randint(0, lithe.K, (lithe.NUM_LEVELS, -(-shape_b[0] * shape_b[1] // 64)))
            lin_.weight += 4.0 * book.decode(idx_b, shape_b, "B") @ book.decode(idx_a, shape_a, "A")
    layers = lithe.inject_lithe(model, ["fc1", "fc2"], rank=rank, alpha=rank, requantize_every=10)
    params = [p for layer in layers.values() for p in (layer.lora_A, layer.lora_B)]
    opt = torch.optim.AdamW(params, lr=lr)
    x_eval = torch.randn(512, d)
    with torch.no_grad():
        y_eval = teacher(x_eval)
        model.eval()
        loss0 = nn.functional.mse_loss(model(x_eval), y_eval).item()
    model.train()
    for step in range(steps):
        x = torch.randn(64, d)
        with torch.no_grad():
            y = teacher(x)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (step % 100 == 0 or step == steps - 1):
            print(f"step {step:4d}  train mse {loss.item():.3e}")
    for layer in layers.values():  # final search on the trained proxies
        layer.requantize()
    model.eval()
    with torch.no_grad():
        y_lithe = model(x_eval)
    loss1 = nn.functional.mse_loss(y_lithe, y_eval).item()

    out_dir = Path(out)
    path = out_dir / "toy.lithe"
    ad = lithe.adapter_from_layers(layers, rank=rank, alpha=rank, base_model="toy")
    size = lithe.save_lithe(path, ad)
    ad2 = lithe.load_lithe(path)
    lora = lithe.to_lora_state_dict(ad2, dtype=torch.float32)
    with torch.no_grad():
        y_dense = dense_forward(model, lora, x_eval)
    report = {
        "eval_mse_before": loss0,
        "eval_mse_after": loss1,
        "lithe_file_bytes": size,
        "num_indices": ad2.num_indices,
        "dense_vs_lithe_max_abs_diff": (y_dense - y_lithe).abs().max().item(),
        "indices_roundtrip_identical": all(
            (ad.indices[n][s] == ad2.indices[n][s]).all() for n in ad.indices for s in ("A", "B")),
    }
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./outputs/cpu_smoke")
    args = ap.parse_args()
    r = run(steps=args.steps, rank=args.rank, seed=args.seed, out=args.out)
    ok = (r["eval_mse_after"] < 0.5 * r["eval_mse_before"] and r["dense_vs_lithe_max_abs_diff"] < 1e-4
          and r["indices_roundtrip_identical"])
    print("CPU SMOKE", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
