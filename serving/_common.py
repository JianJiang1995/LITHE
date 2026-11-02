"""Shared helpers for the serving benchmarks (FLUX.1-dev, one GPU).

Adapters are ``.lithe`` files (``lithe.fileio``). The released adapters use the
Kohya-304 module layout; on the diffusers transformer they cover 494 linears
(``lithe.kohya``). Dense LoRA baselines are read from ``.safetensors`` files in
PEFT or Kohya key format, or decoded from the same ``.lithe`` files.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lithe  # noqa: E402
from lithe import codebook as cb  # noqa: E402
from lithe import kohya  # noqa: E402
from lithe.kernels import (IndexLoRALinear, PackedLayout, PerSampleLoRALinear,  # noqa: E402
                           decode_packed, pack_adapter, split_decoded, wrap_linears)

DEFAULT_PROMPT = "a photo of a dog on a beach at sunset"


# ---------------------------------------------------------------------------
# CLI, logging, results
# ---------------------------------------------------------------------------
def base_parser(doc: str, flux: bool = True, adapters_required: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    if flux:
        ap.add_argument("--pretrained", default=os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
                        help="FLUX.1-dev diffusers folder or hub id (default: $FLUX_MODEL_PATH)")
    ap.add_argument("--adapter-dir", required=adapters_required, help="folder searched recursively for *.lithe files")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    ap.add_argument("--out", default=None, help="write results JSON here")
    return ap


class Log:
    def __init__(self):
        self.lines: List[str] = []

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.lines.append(line)


def save_json(path: Optional[str], obj) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
    print(f"saved -> {p}", flush=True)


def stats_ms(values: Sequence[float]) -> Dict[str, float]:
    v = sorted(values)
    n = len(v)
    return {"n": n, "mean": statistics.mean(v), "std": statistics.pstdev(v) if n > 1 else 0.0,
            "p50": v[n // 2], "p95": v[max(0, int(n * 0.95) - 1)], "p99": v[max(0, int(n * 0.99) - 1)],
            "min": v[0], "max": v[-1]}


def setup_device(gpu: int) -> str:
    if not torch.cuda.is_available():
        raise SystemExit("these benchmarks need a CUDA GPU")
    torch.cuda.set_device(gpu)
    return f"cuda:{gpu}"


def vram_alloc_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9


def vram_reserved_gb() -> float:
    return torch.cuda.memory_reserved() / 1e9


def vram_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def env_info() -> Dict[str, str]:
    info = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = "unavailable"
    try:
        import diffusers
        info["diffusers"] = diffusers.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
def list_adapters(adapter_dir: str, n: Optional[int] = None) -> List[Path]:
    files = sorted(Path(adapter_dir).rglob("*.lithe"))
    if not files:
        raise SystemExit(f"no .lithe files under {adapter_dir}")
    return files[:n] if n else files


def load_adapters(paths: Sequence[Path]) -> List["lithe.LitheAdapter"]:
    ads = [lithe.load_lithe(p) for p in paths]
    names = {m["name"] for m in ads[0].modules}
    for p, ad in zip(paths, ads):
        if {m["name"] for m in ad.modules} != names or ad.rank != ads[0].rank:
            raise SystemExit(f"{p}: module set or rank differs from {paths[0]}")
    return ads


def adapter_prompt(ad, fallback: str = DEFAULT_PROMPT) -> str:
    return ad.metadata.get("example_prompt") or fallback


def adapter_label(path: Path) -> str:
    return path.stem


def pools(device, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    pa, pb = cb.twin_pools()
    return pa.to(device=device, dtype=dtype).contiguous(), pb.to(device=device, dtype=dtype).contiguous()


def decoded_diffusers_factors(ad, device, dtype=torch.bfloat16) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Per-module decode (PyTorch path) -> {diffusers path: (A, B, scale)} in ``dtype``."""
    fac = lithe.decode_factors(ad, device=device)
    scale = float(ad.alpha) / ad.rank
    if ad.key_format == "kohya":
        fac = kohya.split_factors(fac)
    else:
        fac = {(k[len("transformer."):] if k.startswith("transformer.") else k): v for k, v in fac.items()}
    return {k: (a.to(dtype).contiguous(), b.to(dtype).contiguous(), scale) for k, (a, b) in fac.items()}


_PEFT = re.compile(r"^(?:base_model\.model\.)?(?:transformer\.)?(.+)\.lora_(A|B)\.weight$")
_KOHYA = re.compile(r"^(lora_unet_.+)\.(lora_down|lora_up)\.weight$")


def load_dense_lora(path, device, dtype=torch.bfloat16) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    """A rank-r LoRA ``.safetensors`` (PEFT or Kohya keys) -> {diffusers path: (A, B, scale)}."""
    from safetensors.torch import load_file
    sd = load_file(str(path))
    peft, koh, alphas = {}, {}, {}
    for k, v in sd.items():
        m = _PEFT.match(k)
        if m:
            peft.setdefault(m.group(1), {})[m.group(2)] = v
            continue
        m = _KOHYA.match(k)
        if m:
            koh.setdefault(m.group(1), {})["A" if m.group(2) == "lora_down" else "B"] = v
        elif k.endswith(".alpha"):
            alphas[k[: -len(".alpha")]] = float(v)
    out = {}
    for name, ab in peft.items():
        out[name] = (ab["A"].to(device, dtype), ab["B"].to(device, dtype), 1.0)
    if koh:
        split = kohya.split_factors({n: (ab["A"], ab["B"]) for n, ab in koh.items()})
        rank_of = {n: ab["A"].shape[0] for n, ab in koh.items()}
        for name, ab in koh.items():
            scale = alphas.get(name, rank_of[name]) / rank_of[name]
            for path, _, _ in kohya.split_spec(name, ab["A"].shape, ab["B"].shape):
                a, b = split[path]
                out[path] = (a.to(device, dtype).contiguous(), b.to(device, dtype).contiguous(), scale)
    if not out:
        raise ValueError(f"{path}: no LoRA keys found")
    return out


def write_dense_lora_from_lithe(ad, path) -> None:
    """Decode an adapter into a standard LoRA file (bf16; Kohya or PEFT keys as the adapter)."""
    from safetensors.torch import save_file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(lithe.to_lora_state_dict(ad), str(path))


# ---------------------------------------------------------------------------
# Bit-packed resident bank (fused decode)
# ---------------------------------------------------------------------------
class PackedBank:
    """GPU-resident 9-bit packed index slots sharing one layout and one pair of fp32 pools."""

    def __init__(self, layout: PackedLayout, device):
        self.layout = layout
        self.device = device
        self.pool_a, self.pool_b = pools(device)
        layout.sides_on(device)
        self.slots: List[torch.Tensor] = []

    @classmethod
    def from_adapters(cls, ads, device) -> Tuple["PackedBank", List[torch.Tensor]]:
        words0, layout = pack_adapter(ads[0])
        bank = cls(layout, device)
        cpu_words = [words0] + [pack_adapter(ad, layout)[0] for ad in ads[1:]]
        return bank, cpu_words

    def add(self, words_cpu: torch.Tensor) -> int:
        self.slots.append(words_cpu.to(self.device).contiguous())
        return len(self.slots) - 1

    def fill_to(self, n: int, template_slot: int = 0) -> None:
        """Grow to ``n`` slots by cloning a resident slot (distinct allocations, same size)."""
        src = self.slots[template_slot]
        while len(self.slots) < n:
            self.slots.append(src.clone())

    def shrink_to(self, n: int) -> None:
        del self.slots[n:]

    def decode(self, slot: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Fused decode of one slot -> {module name: (A, B)} bf16 views."""
        out = decode_packed(self.slots[slot], self.layout, self.pool_a, self.pool_b)
        return split_decoded(out, self.layout)

    def decode_diffusers(self, slot: int, key_format: str = "kohya") -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
        fac = self.decode(slot)
        if key_format == "kohya":
            fac = kohya.split_factors(fac)
        s = self.layout.scaling
        return {k: (a, b, s) for k, (a, b) in fac.items()}

    @property
    def slot_bytes(self) -> int:
        return self.layout.num_words * 8


# ---------------------------------------------------------------------------
# Wrapping the FLUX transformer
# ---------------------------------------------------------------------------
def load_pipe(pretrained: str, device):
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(pretrained, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def wrap_decoded(transformer, paths: Sequence[str], kernel: str) -> Dict[str, PerSampleLoRALinear]:
    return wrap_linears(transformer, paths, lambda p, lin: PerSampleLoRALinear(lin, kernel))


def install_per_sample(wrapped: Dict[str, PerSampleLoRALinear], decoded: Sequence[Dict]) -> None:
    """Point every wrapped linear at the factors of ``decoded[i]`` for slot i (no copy)."""
    for path, mod in wrapped.items():
        mod.set_factors([d[path] for d in decoded], per_sample=True)


def install_single(wrapped: Dict[str, PerSampleLoRALinear], decoded: Dict) -> None:
    for path, mod in wrapped.items():
        mod.set_factors([decoded[path]], per_sample=False)


def select_paths(paths: Sequence[str], module_set: str = "all") -> List[str]:
    """``all``: every linear the adapter covers (494 for Kohya-304); ``attn-ff``: only the
    342 attention / feed-forward linears of ``lithe.FLUX_TARGET_MODULES`` (the diffusers
    module set of the adapters behind the paper's wall-time table)."""
    if module_set == "all":
        return list(paths)
    if module_set == "attn-ff":
        return [p for p in paths if any(p.endswith(s) for s in lithe.FLUX_TARGET_MODULES)]
    raise ValueError(f"unknown module set {module_set!r}")


def wrap_index(transformer, ads, device, dtype=torch.bfloat16,
               module_set: str = "all") -> Dict[str, IndexLoRALinear]:
    """Index-resident wrappers: every linear holds the index slots of all ``ads``."""
    per_ad = [kohya.split_indices(ad) for ad in ads]
    pa, pb = pools(device, dtype)
    scale = float(ads[0].alpha) / ads[0].rank

    def factory(path, lin):
        ia = torch.stack([torch.from_numpy(s[path][0].astype(np.int32)) for s in per_ad]).to(device)
        ib = torch.stack([torch.from_numpy(s[path][1].astype(np.int32)) for s in per_ad]).to(device)
        return IndexLoRALinear(lin, ia, ib, pa, pb, ads[0].rank, scale)

    return wrap_linears(transformer, select_paths(list(per_ad[0].keys()), module_set), factory)


def set_assign(wrapped: Dict, assign: Optional[torch.Tensor]) -> None:
    for mod in wrapped.values():
        mod.assign = assign


def render(pipe, prompts: Sequence[str], size: int, steps: int, guidance: float = 3.5, seed: int = 42):
    """One batched pipeline call; returns (images, wall seconds)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t0 = sync_time()
    with torch.no_grad():
        images = pipe(prompt=list(prompts), height=size, width=size, num_inference_steps=steps,
                      guidance_scale=guidance, max_sequence_length=128, generator=g).images
    return images, sync_time() - t0
