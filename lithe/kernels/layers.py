"""Linear wrappers that route every sample of a batch to its own adapter.

``PerSampleLoRALinear``  decoded factors (A, B, scale) per adapter slot; the
                         delta runs either as a per-sample Python loop
                         (``kernel="loop"``) or through the two-stage Triton
                         SGMV kernel (``kernel="sgmv"``).
``IndexLoRALinear``      index-resident slots; the delta is decoded inside the
                         shrink / expand tiles (``lithe.kernels.shared``).

Routing: with ``assign`` unset, sample ``i`` uses slot ``i % n_slots`` (the
convention of the original benchmark scripts); otherwise ``assign`` (B,) gives
the slot of every sample.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .sgmv import per_sample_lora_delta, per_sample_lora_delta_reference
from .shared import shared_codebook_lora_delta

Factor = Tuple[torch.Tensor, torch.Tensor, float]


class PerSampleLoRALinear(nn.Module):
    """y = W x + scale_j * B_j A_j x for the slot j of every sample (decoded factors)."""

    def __init__(self, base: nn.Linear, kernel: str = "sgmv"):
        super().__init__()
        if kernel not in ("sgmv", "loop"):
            raise ValueError(f"kernel must be 'sgmv' or 'loop', got {kernel!r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.kernel = kernel
        self.factors: List[Factor] = []
        self.per_sample = False
        self.assign: Optional[torch.Tensor] = None
        self._stacked = None

    def set_factors(self, factors: Sequence[Factor], per_sample: bool = True) -> None:
        """Install slot factors by reference (no copy); stacking happens lazily in forward."""
        self.factors = list(factors)
        self.per_sample = per_sample
        self._stacked = None

    def clear(self) -> None:
        self.set_factors([], per_sample=False)

    def _slots(self, bsz: int, device) -> torch.Tensor:
        if self.assign is not None:
            return self.assign.to(device=device, dtype=torch.long)
        return torch.arange(bsz, device=device) % len(self.factors)

    def _build_stacked(self):
        if self._stacked is None:
            a = torch.stack([f[0] for f in self.factors]).contiguous()
            b = torch.stack([f[1] for f in self.factors]).contiguous()
            s = torch.tensor([float(f[2]) for f in self.factors], dtype=a.dtype, device=a.device)
            self._stacked = (a, b, s)
        return self._stacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.factors:
            return out
        if not self.per_sample:
            a, b, s = self.factors[0]
            return out + F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * s
        bsz = x.shape[0]
        if self.kernel == "loop":
            delta = torch.zeros_like(out)
            # host-side slot list: no device sync per call
            n = len(self.factors)
            slots = [i % n for i in range(bsz)] if self.assign is None else self.assign.tolist()
            for i, j in enumerate(slots):
                a, b, s = self.factors[j]
                delta[i:i + 1] = F.linear(F.linear(x[i:i + 1], a.to(x.dtype)), b.to(x.dtype)) * s
            return out + delta
        a, b, s = self._build_stacked()
        if bsz != a.shape[0] or self.assign is not None:
            idx = self._slots(bsz, x.device)
            a, b, s = a.index_select(0, idx), b.index_select(0, idx), s.index_select(0, idx)
        a, b, s = a.to(x.dtype), b.to(x.dtype), s.to(x.dtype)
        if x.dim() == 3:
            delta = per_sample_lora_delta(x, a, b, s)
        elif x.dim() == 2:      # modulation linears see (B, d); bmm is enough here
            delta = per_sample_lora_delta_reference(x.unsqueeze(1), a, b, s).squeeze(1)
        else:
            raise ValueError(f"unsupported input rank {x.dim()}")
        return out + delta


class IndexLoRALinear(nn.Module):
    """y = W x + per-sample delta decoded on the fly from resident index slots."""

    def __init__(self, base: nn.Linear, idx_a: torch.Tensor, idx_b: torch.Tensor,
                 pool_a: torch.Tensor, pool_b: torch.Tensor, rank: int, scaling: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        # (n_slots, L, n_blocks) int32; pools (L, K, block) in the activation dtype
        self.register_buffer("idx_a", idx_a.to(torch.int32), persistent=False)
        self.register_buffer("idx_b", idx_b.to(torch.int32), persistent=False)
        self.register_buffer("pool_a", pool_a, persistent=False)
        self.register_buffer("pool_b", pool_b, persistent=False)
        self.rank = int(rank)
        self.d_out = base.out_features
        self.scaling = float(scaling)
        self.assign: Optional[torch.Tensor] = None

    @property
    def num_slots(self) -> int:
        return int(self.idx_a.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        squeeze = x.dim() == 2
        x3 = x.unsqueeze(1) if squeeze else x
        bsz = x3.shape[0]
        assign = self.assign
        if assign is None:
            assign = torch.zeros(bsz, dtype=torch.int32, device=x.device)
        delta = shared_codebook_lora_delta(x3, self.idx_a, self.pool_a, self.idx_b, self.pool_b,
                                           assign, self.rank, self.d_out, self.scaling,
                                           out_dtype=x.dtype)
        return out + (delta.squeeze(1) if squeeze else delta)


def wrap_linears(model: nn.Module, paths: Iterable[str],
                 factory: Callable[[str, nn.Linear], nn.Module]) -> Dict[str, nn.Module]:
    """Replace ``model.<path>`` (an ``nn.Linear``) by ``factory(path, linear)`` for every path."""
    wrapped = {}
    for path in paths:
        base = model.get_submodule(path)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{path} is not nn.Linear")
        parent_name, _, child = path.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        module = factory(path, base)
        setattr(parent, child, module)
        wrapped[path] = module
    return wrapped
