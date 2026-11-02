"""Index-resident multi-adapter serving: heterogeneous adapters in one forward.

Every wrapped linear keeps only the uint16 index streams of all loaded
adapters. For each forward it decodes the adapters that the current batch
uses and applies, per sample i, the rank-stacked update

    A_i = vcat_a[ sqrt(w_ia) A_a ],  B_i = hcat_a[ sqrt(w_ia) B_a ],  delta_i = B_i A_i x_i

so one sample can use one adapter, a weighted mixture, or none (empty route).
This is a readable PyTorch reference of the routing math; the paper's fused
SGMV kernel is not part of this release.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from . import codebook as cb
from . import quant
from .fileio import LitheAdapter

Route = Sequence[Tuple[int, float]]  # [(adapter_id, weight), ...] for one sample


class Router:
    """Per-batch routing table shared by all MultiAdapterLinear layers."""

    def __init__(self):
        self.ids: Optional[torch.Tensor] = None      # (b, m) int64, -1 = padding
        self.weights: Optional[torch.Tensor] = None  # (b, m) float32

    def set(self, routes: Sequence[Route]) -> None:
        m = max(1, max((len(r) for r in routes), default=1))
        ids = torch.full((len(routes), m), -1, dtype=torch.int64)
        w = torch.zeros(len(routes), m, dtype=torch.float32)
        for i, r in enumerate(routes):
            for j, (a, wt) in enumerate(r):
                if wt < 0:
                    raise ValueError("mixture weights must be non-negative")
                ids[i, j], w[i, j] = int(a), float(wt)
        self.ids, self.weights = ids, w


class MultiAdapterLinear(nn.Module):
    def __init__(self, base: nn.Linear, router: Router, rank: int,
                 pool_a: torch.Tensor, pool_b: torch.Tensor):
        super().__init__()
        self.base = base
        self.router = router
        self.rank = rank
        self.shape_a = (rank, base.in_features)
        self.shape_b = (base.out_features, rank)
        self.register_buffer("pool_a", pool_a, persistent=False)
        self.register_buffer("pool_b", pool_b, persistent=False)
        # index banks: (n_adapters, L, n_blocks) int16 (K = 512 fits), plus per-adapter alpha / r
        self.register_buffer("idx_a", torch.zeros(0, pool_a.shape[0], quant.num_blocks(self.shape_a), dtype=torch.int16), persistent=False)
        self.register_buffer("idx_b", torch.zeros(0, pool_b.shape[0], quant.num_blocks(self.shape_b), dtype=torch.int16), persistent=False)
        self.register_buffer("scales", torch.zeros(0), persistent=False)

    def add_adapter(self, idx_a: np.ndarray, idx_b: np.ndarray, scale: float) -> None:
        dev = self.idx_a.device
        self.idx_a = torch.cat([self.idx_a, torch.from_numpy(idx_a.astype(np.int16))[None].to(dev)])
        self.idx_b = torch.cat([self.idx_b, torch.from_numpy(idx_b.astype(np.int16))[None].to(dev)])
        self.scales = torch.cat([self.scales, torch.tensor([scale], device=dev)])

    def _decode(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode adapters ``ids`` (u,) -> A (u, r, d_in), B (u, d_out, r) float32."""
        a = torch.stack([quant.decode(self.idx_a[i], self.shape_a, self.pool_a) for i in ids.tolist()])
        b = torch.stack([quant.decode(self.idx_b[i], self.shape_b, self.pool_b) for i in ids.tolist()])
        return a, b * self.scales[ids].view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        ids, w = self.router.ids, self.router.weights
        if ids is None or (ids < 0).all():
            return out
        if ids.shape[0] != x.shape[0]:
            raise ValueError(f"route batch {ids.shape[0]} != input batch {x.shape[0]}")
        ids, w = ids.to(x.device), w.to(x.device)
        active = torch.unique(ids[ids >= 0])
        a_u, b_u = self._decode(active)
        lut = torch.full((int(active.max()) + 1,), 0, dtype=torch.int64, device=x.device)
        lut[active] = torch.arange(active.numel(), device=x.device)
        valid = ids >= 0
        sel = lut[ids.clamp(min=0)]                                    # (b, m)
        g = (w * valid).clamp(min=0).sqrt()                            # (b, m)
        a_s = a_u[sel] * g[..., None, None]                            # (b, m, r, d_in)
        b_s = b_u[sel] * g[..., None, None]                            # (b, m, d_out, r)
        bsz, m = sel.shape
        a_s = a_s.reshape(bsz, m * self.rank, -1)                      # vcat along rank
        b_s = b_s.permute(0, 2, 1, 3).reshape(bsz, b_s.shape[2], m * self.rank)  # hcat along rank
        xs = x.reshape(bsz, -1, x.shape[-1])
        delta = torch.bmm(torch.bmm(xs, a_s.transpose(1, 2).to(x.dtype)), b_s.transpose(1, 2).to(x.dtype))
        return out + delta.reshape(out.shape)


def attach_adapters(model: nn.Module, adapters: List[LitheAdapter],
                    router: Optional[Router] = None) -> Tuple[Router, Dict[str, MultiAdapterLinear]]:
    """Wrap every linear named in the adapters' module list and load all index streams."""
    if not adapters:
        raise ValueError("need at least one adapter")
    router = router or Router()
    ref = adapters[0]
    names = [mm["name"] for mm in ref.modules]
    for ad in adapters[1:]:
        if [mm["name"] for mm in ad.modules] != names or ad.rank != ref.rank:
            raise ValueError("all adapters must cover the same modules with the same rank")
    pool_a = cb.build_pools(ref.sigma_a, ref.block_size, ref.n_bits)
    pool_b = cb.build_pools(ref.sigma_b, ref.block_size, ref.n_bits)
    layers: Dict[str, MultiAdapterLinear] = {}
    for name in names:
        path = name[len("transformer."):] if name.startswith("transformer.") else name
        base = model.get_submodule(path)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{path} is not nn.Linear")
        parent_name, _, child = path.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = MultiAdapterLinear(base, router, ref.rank, pool_a, pool_b).to(base.weight.device)
        for ad in adapters:
            layer.add_adapter(ad.indices[name]["A"], ad.indices[name]["B"], float(ad.alpha) / ad.rank)
        setattr(parent, child, layer)
        layers[name] = layer
    return router, layers


def parse_route(spec: str) -> List[Tuple[int, float]]:
    """'0' -> [(0, 1.0)];  '0:0.6,2:0.4' -> [(0, 0.6), (2, 0.4)];  'none' -> []."""
    if spec.strip().lower() in ("", "none", "-"):
        return []
    route = []
    for part in spec.split(","):
        a, _, wt = part.partition(":")
        route.append((int(a), float(wt) if wt else 1.0))
    return route
