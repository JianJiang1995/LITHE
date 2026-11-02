"""Optional Triton import shared by the kernel modules."""
from __future__ import annotations

import torch

try:  # Triton is optional: every kernel has a PyTorch reference.
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the environment
    triton = None
    tl = None
    HAS_TRITON = False


def use_triton(*tensors: torch.Tensor) -> bool:
    """True when Triton is importable and every tensor lives on a CUDA device."""
    return HAS_TRITON and all(t.is_cuda for t in tensors)
