"""LITHE LoRA linear wrapper and injection into a diffusers model."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

from . import codebook as cb
from . import quant

# FLUX.1-dev (diffusers FluxTransformer2DModel) attention + FFN linears.
# Suffix match over module paths gives 19 x 12 + 38 x 3 = 342 linears.
FLUX_TARGET_MODULES: List[str] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


class TwinCodebook(nn.Module):
    """Holds the deterministic pools as non-persistent buffers (never saved)."""

    def __init__(self, block_size: int = cb.BLOCK_SIZE, n_bits: int = cb.N_BITS,
                 sigma_a: Sequence[float] = cb.SIGMA_A, sigma_b: Sequence[float] = cb.SIGMA_B):
        super().__init__()
        pool_a, pool_b = cb.twin_pools(block_size, n_bits, sigma_a, sigma_b)
        self.block_size = block_size
        self.n_bits = n_bits
        self.sigma_a = tuple(float(s) for s in sigma_a)
        self.sigma_b = tuple(float(s) for s in sigma_b)
        self.register_buffer("pool_a", pool_a, persistent=False)
        self.register_buffer("pool_b", pool_b, persistent=False)
        self.register_buffer("proj", cb.projection_matrix(block_size, n_bits), persistent=False)
        self._per_device: Dict[str, tuple] = {}

    def tensors(self, device) -> tuple:
        """(pool_a, pool_b, proj) on ``device``; copied once per device."""
        key = str(device)
        if key not in self._per_device:
            self._per_device[key] = (self.pool_a.to(device), self.pool_b.to(device), self.proj.to(device))
        return self._per_device[key]

    def encode(self, w: torch.Tensor, side: str):
        pool_a, pool_b, proj = self.tensors(w.device)
        return quant.encode(w, pool_a if side == "A" else pool_b, proj)

    def decode(self, indices: torch.Tensor, shape, side: str) -> torch.Tensor:
        pool_a, pool_b, _ = self.tensors(indices.device)
        return quant.decode(indices, shape, pool_a if side == "A" else pool_b)


class LitheLinear(nn.Module):
    """y = W x + (B_hat A_hat x) * alpha / r with A_hat, B_hat on the Hadamard lattice.

    Training: fp32 proxies ``lora_A`` (r, d_in) and ``lora_B`` (d_out, r) are
    re-quantised every ``requantize_every`` forwards (optionally earlier when
    the L2 drift since the last search exceeds ``drift_tau``). Between searches
    the cached reconstruction is reused; gradients reach the proxies through
    the straight-through expression in every forward.

    Inference: after ``load_indices`` the module only holds decoded factors.
    """

    def __init__(self, base: nn.Linear, codebook: TwinCodebook, rank: int = 16,
                 alpha: Optional[float] = None, requantize_every: int = 10,
                 drift_tau: Optional[float] = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / rank
        self.requantize_every = max(1, int(requantize_every))
        self.drift_tau = drift_tau
        # Not registered as a submodule: one codebook is shared by all wrappers.
        self.__dict__["codebook"] = codebook
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._forward_count = 0
        self._cache_A: Optional[torch.Tensor] = None
        self._cache_B: Optional[torch.Tensor] = None
        self._last_A: Optional[torch.Tensor] = None
        self._last_B: Optional[torch.Tensor] = None
        self._frozen = False

    # ---- quantisation -------------------------------------------------
    @torch.no_grad()
    def requantize(self) -> None:
        _, self._cache_A = self.codebook.encode(self.lora_A, "A")
        _, self._cache_B = self.codebook.encode(self.lora_B, "B")
        self._last_A = self.lora_A.detach().clone()
        self._last_B = self.lora_B.detach().clone()

    def _need_requantize(self) -> bool:
        if self._cache_A is None:
            return True  # pre-quantise init on the first forward
        if self._forward_count % self.requantize_every == 0:
            return True
        if self.drift_tau is not None:
            drift = max((self.lora_A - self._last_A).norm().item(),
                        (self.lora_B - self._last_B).norm().item())
            return drift > self.drift_tau
        return False

    @torch.no_grad()
    def indices(self) -> Dict[str, torch.Tensor]:
        """Fresh search on the current proxies: {'A': (L, nA), 'B': (L, nB)} int64."""
        idx_a, _ = self.codebook.encode(self.lora_A, "A")
        idx_b, _ = self.codebook.encode(self.lora_B, "B")
        return {"A": idx_a, "B": idx_b}

    @torch.no_grad()
    def load_indices(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> None:
        """Freeze the module to the factors decoded from index streams."""
        dev = self.lora_A.device
        self._cache_A = self.codebook.decode(idx_a.to(dev), self.lora_A.shape, "A")
        self._cache_B = self.codebook.decode(idx_b.to(dev), self.lora_B.shape, "B")
        self._frozen = True
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)

    def decoded_factors(self):
        """(A_hat, B_hat) float32 as used by the forward pass."""
        if self._cache_A is None:
            self.requantize()
        return self._cache_A, self._cache_B

    # ---- forward ------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._frozen or not self.training:
            a_hat, b_hat = self.decoded_factors()
        else:
            if self._need_requantize():
                self.requantize()
            self._forward_count += 1
            a_hat = quant.ste(self.lora_A, self._cache_A)
            b_hat = quant.ste(self.lora_B, self._cache_B)
        delta = nn.functional.linear(nn.functional.linear(x, a_hat.to(x.dtype)), b_hat.to(x.dtype))
        return out + delta * self.scaling


def _match(path: str, targets: Iterable[str]) -> bool:
    return any(path == t or path.endswith("." + t) for t in targets)


def inject_lithe(model: nn.Module, target_modules: Sequence[str] = FLUX_TARGET_MODULES,
                 rank: int = 16, alpha: Optional[float] = None, requantize_every: int = 10,
                 drift_tau: Optional[float] = None,
                 codebook: Optional[TwinCodebook] = None) -> Dict[str, LitheLinear]:
    """Replace every nn.Linear whose path ends with a target suffix by a LitheLinear.

    Returns an ordered {module_path: LitheLinear}; this order defines the
    layout of the index stream.
    """
    codebook = codebook if codebook is not None else TwinCodebook()
    targets = [(name, mod) for name, mod in model.named_modules()
               if isinstance(mod, nn.Linear) and _match(name, target_modules)]
    if not targets:
        raise ValueError("no target linear found; check target_modules")
    wrapped: Dict[str, LitheLinear] = {}
    for name, mod in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = LitheLinear(mod, codebook, rank=rank, alpha=alpha,
                            requantize_every=requantize_every, drift_tau=drift_tau)
        setattr(parent, child, layer)
        wrapped[name] = layer
    return wrapped


def lithe_layers(model: nn.Module) -> Dict[str, LitheLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LitheLinear)}
