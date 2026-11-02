"""Conversions between ``.lithe`` index files and dense LoRA state dicts."""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from . import codebook as cb
from . import quant
from .fileio import LitheAdapter


def adapter_pools(ad: LitheAdapter) -> Tuple[torch.Tensor, torch.Tensor]:
    return (cb.build_pools(ad.sigma_a, ad.block_size, ad.n_bits),
            cb.build_pools(ad.sigma_b, ad.block_size, ad.n_bits))


def decode_factors(ad: LitheAdapter, device="cpu") -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """name -> (A_hat (r, d_in), B_hat (d_out, r)), float32, closed-form decode."""
    pool_a, pool_b = (p.to(device) for p in adapter_pools(ad))
    out = {}
    for m in ad.modules:
        idx = ad.indices[m["name"]]
        a = quant.decode(torch.from_numpy(idx["A"].astype(np.int64)).to(device), m["shape_a"], pool_a)
        b = quant.decode(torch.from_numpy(idx["B"].astype(np.int64)).to(device), m["shape_b"], pool_b)
        out[m["name"]] = (a, b)
    return out


def to_lora_state_dict(ad: LitheAdapter, dtype: torch.dtype = torch.bfloat16) -> Dict[str, torch.Tensor]:
    """Dense LoRA state dict loadable by ``FluxPipeline.load_lora_weights``.

    The factor alpha / r is folded into B so the file does not depend on how a
    loader treats a missing or present alpha. Diffusers-named adapters are
    written with PEFT keys (``transformer.<path>.lora_A.weight``); Kohya-named
    adapters keep Kohya keys (``<name>.lora_down.weight`` + ``.alpha``), which
    diffusers converts on load.
    """
    scale = float(ad.alpha) / ad.rank
    sd: Dict[str, torch.Tensor] = {}
    for name, (a, b) in decode_factors(ad).items():
        b = b * scale
        if ad.key_format == "diffusers":
            prefix = name if name.startswith("transformer.") else f"transformer.{name}"
            sd[f"{prefix}.lora_A.weight"] = a.to(dtype).contiguous()
            sd[f"{prefix}.lora_B.weight"] = b.to(dtype).contiguous()
        elif ad.key_format == "kohya":
            sd[f"{name}.lora_down.weight"] = a.to(dtype).contiguous()
            sd[f"{name}.lora_up.weight"] = b.to(dtype).contiguous()
            sd[f"{name}.alpha"] = torch.tensor(float(ad.rank), dtype=dtype)
        else:
            raise ValueError(f"unknown key_format {ad.key_format!r}")
    return sd


_PEFT_KEY = re.compile(r"^(?:base_model\.model\.)?(?:transformer\.)?(.+)\.lora_(A|B)\.weight$")


def from_dense_lora(state_dict: Dict[str, torch.Tensor], rank: Optional[int] = None,
                    base_model: str = "black-forest-labs/FLUX.1-dev") -> LitheAdapter:
    """Post-hoc twin-RVQ encoding of a continuous PEFT/diffusers LoRA.

    This is the post-training quantisation route. The paper's adapters are
    trained natively with STE (``scripts/train_lithe_flux.py``); post-hoc
    encoding of a LoRA trained without the codebook is generally lossier.
    """
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    alphas: Dict[str, float] = {}
    for key, value in state_dict.items():
        m = _PEFT_KEY.match(key)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = value.float()
        elif key.endswith(".alpha"):
            alphas[re.sub(r"^(?:transformer\.)", "", key[: -len(".alpha")])] = float(value)
    if not pairs:
        raise ValueError("no PEFT-style lora_A / lora_B keys found")
    pool_a, pool_b = cb.twin_pools()
    proj = cb.projection_matrix()
    modules, indices = [], {}
    r_seen = None
    for name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            raise ValueError(f"{name}: missing lora_A or lora_B")
        a, b = ab["A"], ab["B"]
        r = a.shape[0]
        if rank is not None and r != rank:
            raise ValueError(f"{name}: rank {r} != {rank}")
        r_seen = r
        if name in alphas:  # fold alpha / r so the encoded adapter uses alpha = r
            b = b * (alphas[name] / r)
        idx_a, _ = quant.encode(a, pool_a, proj)
        idx_b, _ = quant.encode(b, pool_b, proj)
        modules.append({"name": name, "shape_a": list(a.shape), "shape_b": list(b.shape)})
        indices[name] = {"A": idx_a.numpy().astype(np.uint16), "B": idx_b.numpy().astype(np.uint16)}
    return LitheAdapter(modules=modules, indices=indices, rank=r_seen, alpha=float(r_seen),
                        key_format="diffusers", base_model=base_model,
                        metadata={"source": "post-hoc encoding of a dense LoRA",
                                  "module_set": f"diffusers-{len(modules)}"})


def from_training_checkpoint(path, check_codebook: bool = True) -> LitheAdapter:
    """Import an index checkpoint written by the original research trainers.

    Two pickled layouts are understood: the diffusers-fork trainer
    (``indices_per_subject`` + ``codebook_pools`` / ``codebook_pools_b``) and the
    Kohya sd-scripts network module (``format == 'lithe_kohya_discrete_v1'``).
    Only load files you trust: this uses ``torch.load(weights_only=False)``.
    With ``check_codebook`` the stored pools are compared with the
    deterministic construction (after rounding to the stored dtype).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") == "lithe_kohya_discrete_v1":
        records, key_format = ck["indices"], "kohya"
        cfg = ck["config"]
        rank_alpha = None
        get_shapes = lambda r: (list(r["shape_A"]), list(r["shape_B"]))  # noqa: E731
    elif "indices_per_subject" in ck:
        records, key_format = ck["indices_per_subject"], "diffusers"
        cfg = ck["config"]
        rank_alpha = (int(cfg["rank"]), float(cfg["alpha"]))
        if cfg.get("codebook_init") != "asym_hadamard9_multiscale_b":
            raise ValueError(f"unsupported codebook_init {cfg.get('codebook_init')!r}")
        get_shapes = lambda r: (list(r["A_shape"]), list(r["B_shape"]))  # noqa: E731
    else:
        raise ValueError(f"{path}: unrecognised checkpoint layout")
    if int(cfg.get("K", cb.K)) != cb.K or int(cfg.get("block_size", cfg.get("B", 64))) != cb.BLOCK_SIZE:
        raise ValueError("only K=512, B=64 checkpoints are supported")
    if check_codebook:
        rel_err = codebook_mismatch(ck["codebook_pools"], ck["codebook_pools_b"])
        if rel_err > 1e-2:
            raise ValueError(f"stored pools differ from the Hadamard construction (rel {rel_err:.2e})")
    modules, indices = [], {}
    for name, rec in records.items():
        shape_a, shape_b = get_shapes(rec)
        to_np = lambda t: (t.numpy() if torch.is_tensor(t) else np.asarray(t)).astype(np.int64)  # noqa: E731
        ia, ib = to_np(rec["A"]), to_np(rec["B"])
        if ia.min() < 0 or ib.min() < 0 or ia.max() >= cb.K or ib.max() >= cb.K:
            raise ValueError(f"{name}: index out of range")
        modules.append({"name": name, "shape_a": shape_a, "shape_b": shape_b})
        indices[name] = {"A": ia.astype(np.uint16), "B": ib.astype(np.uint16)}
    rank = rank_alpha[0] if rank_alpha else modules[0]["shape_a"][0]
    alpha = rank_alpha[1] if rank_alpha else float(rank)
    module_set = f"{key_format}-{len(modules)}"   # e.g. "kohya-304" or "diffusers-342"
    return LitheAdapter(modules=modules, indices=indices, rank=rank, alpha=alpha,
                        key_format=key_format,
                        metadata={"source": "imported training checkpoint", "module_set": module_set})


def codebook_mismatch(pools_a, pools_b) -> float:
    """Max relative difference (max |stored - ref| / max |ref|) over all stored pools.

    Original trainers stored pools in fp32 or bf16, so a match is at the level
    of the stored dtype (about 1e-5 for fp32, 4e-3 for bf16); a wrong sign
    convention or scale gives an error of order 1.
    """
    ref_a, ref_b = cb.twin_pools()
    err = 0.0
    for stored, ref in zip(list(pools_a) + list(pools_b), list(ref_a) + list(ref_b)):
        stored = stored.detach().cpu().float()
        err = max(err, ((stored - ref).abs().max() / ref.abs().max()).item())
    return err
