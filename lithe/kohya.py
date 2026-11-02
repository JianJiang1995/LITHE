"""Kohya (BFL-style fused) FLUX.1-dev module names -> diffusers ``FluxTransformer2DModel`` linears.

The released adapters use the Kohya sd-scripts layout: 304 linears per adapter,
19 double blocks x 10 + 38 single blocks x 3, with fused QKV projections and
the modulation layers. The diffusers transformer splits the fused projections,
so one Kohya module maps to one or more diffusers linears that share the same
A factor and take consecutive row ranges of B (494 linears in total):

    double_blocks_i_img_attn_qkv   -> transformer_blocks.i.attn.{to_q, to_k, to_v}
    double_blocks_i_txt_attn_qkv   -> transformer_blocks.i.attn.{add_q_proj, add_k_proj, add_v_proj}
    double_blocks_i_img_attn_proj  -> transformer_blocks.i.attn.to_out.0
    double_blocks_i_txt_attn_proj  -> transformer_blocks.i.attn.to_add_out
    double_blocks_i_{img,txt}_mlp_{0,2} -> transformer_blocks.i.{ff,ff_context}.net.{0.proj,2}
    double_blocks_i_{img,txt}_mod_lin   -> transformer_blocks.i.{norm1,norm1_context}.linear
    single_blocks_i_linear1        -> single_transformer_blocks.i.attn.{to_q, to_k, to_v} + .proj_mlp
    single_blocks_i_linear2        -> single_transformer_blocks.i.proj_out
    single_blocks_i_modulation_lin -> single_transformer_blocks.i.norm.linear

Because B is flattened row-major into 64-value blocks and r * d is a multiple
of 64 for every split point, the split can be done on the index streams as
well as on decoded factors.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Split = List[Tuple[str, int, int]]   # [(diffusers path, first B row, end B row)]

_DOUBLE = {
    "img_attn_proj": "attn.to_out.0",
    "txt_attn_proj": "attn.to_add_out",
    "img_mlp_0": "ff.net.0.proj",
    "img_mlp_2": "ff.net.2",
    "txt_mlp_0": "ff_context.net.0.proj",
    "txt_mlp_2": "ff_context.net.2",
    "img_mod_lin": "norm1.linear",
    "txt_mod_lin": "norm1_context.linear",
}
_QKV = {
    "img_attn_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "txt_attn_qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}
_SINGLE = {"linear2": "proj_out", "modulation_lin": "norm.linear"}


def split_spec(name: str, shape_a, shape_b) -> Split:
    """Diffusers targets of one Kohya module and the B rows each one takes."""
    d_out, d_in = int(shape_b[0]), int(shape_a[1])
    key = name[len("lora_unet_"):] if name.startswith("lora_unet_") else name
    if key.startswith("double_blocks_"):
        blk, _, tail = key[len("double_blocks_"):].partition("_")
        base = f"transformer_blocks.{blk}"
        if tail in _QKV:
            d = d_out // 3
            return [(f"{base}.{sub}", i * d, (i + 1) * d) for i, sub in enumerate(_QKV[tail])]
        if tail in _DOUBLE:
            return [(f"{base}.{_DOUBLE[tail]}", 0, d_out)]
    elif key.startswith("single_blocks_"):
        blk, _, tail = key[len("single_blocks_"):].partition("_")
        base = f"single_transformer_blocks.{blk}"
        if tail == "linear1":   # [q; k; v; mlp], attention width = hidden size = d_in
            d = d_in
            spec = [(f"{base}.attn.{sub}", i * d, (i + 1) * d) for i, sub in enumerate(("to_q", "to_k", "to_v"))]
            return spec + [(f"{base}.proj_mlp", 3 * d, d_out)]
        if tail in _SINGLE:
            return [(f"{base}.{_SINGLE[tail]}", 0, d_out)]
    raise KeyError(f"not a Kohya FLUX.1-dev module name: {name}")


def diffusers_paths(ad) -> List[str]:
    """All diffusers linear paths an adapter touches (494 for the Kohya-304 layout)."""
    if ad.key_format != "kohya":
        return [m["name"][len("transformer."):] if m["name"].startswith("transformer.") else m["name"]
                for m in ad.modules]
    return [p for m in ad.modules for p, _, _ in split_spec(m["name"], m["shape_a"], m["shape_b"])]


def split_factors(factors: Dict[str, tuple]) -> Dict[str, tuple]:
    """{kohya name: (A, B)} -> {diffusers path: (A, B rows)}; B slices of fused modules
    are made contiguous, unsplit modules keep their (view) tensors."""
    out = {}
    for name, (a, b) in factors.items():
        spec = split_spec(name, a.shape, b.shape)
        if len(spec) == 1:
            out[spec[0][0]] = (a, b)
        else:
            for path, r0, r1 in spec:
                out[path] = (a, b[r0:r1].contiguous())
    return out


def split_indices(ad, block_size: int = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, tuple, tuple]]:
    """Index-level split: {diffusers path: (idx_A (L, nA), idx_B (L, nB), shape_a, shape_b)}.

    Works for both key formats; diffusers-format adapters pass through unchanged.
    """
    bs = block_size or ad.block_size
    out = {}
    for m in ad.modules:
        idx = ad.indices[m["name"]]
        sa, sb = tuple(m["shape_a"]), tuple(m["shape_b"])
        if ad.key_format != "kohya":
            path = m["name"][len("transformer."):] if m["name"].startswith("transformer.") else m["name"]
            out[path] = (idx["A"], idx["B"], sa, sb)
            continue
        r = sb[1]
        spec = split_spec(m["name"], sa, sb)
        if len(spec) == 1:
            out[spec[0][0]] = (idx["A"], idx["B"], sa, sb)
            continue
        for path, r0, r1 in spec:
            if (r0 * r) % bs or (r1 * r) % bs:
                raise ValueError(f"{m['name']}: split at rows {r0}:{r1} is not block aligned")
            b0, b1 = r0 * r // bs, r1 * r // bs
            out[path] = (idx["A"], idx["B"][:, b0:b1], sa, (r1 - r0, r))
    return out
