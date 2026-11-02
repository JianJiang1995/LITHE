"""Twin residual vector quantisation (RVQ) against the Hadamard pools.

A LoRA factor of any shape is flattened row-major, zero-padded to a multiple
of the block size and viewed as (n_blocks, B). Each block is encoded with L
greedy residual levels; level l picks the codeword of ``pools[l]`` nearest to
the current residual.

Because all codewords in a pool have the same norm, the nearest codeword is
the one maximising <residual, c_k>, i.e. bit q of the index is
[ <residual, U[:, q]> > 0 ]. This closed form is exact (it equals the argmin
of the Euclidean distance up to ties at exactly zero projection) and never
materialises a K x B distance matrix.
"""
from __future__ import annotations

from typing import Tuple

import torch

from . import codebook as cb


def blockify(w: torch.Tensor, block_size: int = cb.BLOCK_SIZE) -> torch.Tensor:
    """Flatten ``w`` row-major and reshape to (n_blocks, block_size) with zero padding."""
    flat = w.reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(-1, block_size)


def unblockify(blocks: torch.Tensor, shape) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= int(s)
    return blocks.reshape(-1)[:n].view(*shape)


def num_blocks(shape, block_size: int = cb.BLOCK_SIZE) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return -(-n // block_size)


def nearest_index(x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Closed-form nearest Hadamard codeword for each row of ``x`` (N, B) -> (N,) int64."""
    ip = x.float() @ proj.to(device=x.device, dtype=torch.float32)
    weights = 1 << torch.arange(proj.shape[1], device=x.device, dtype=torch.int64)
    return ((ip > 0).to(torch.int64) * weights).sum(dim=1)


@torch.no_grad()
def rvq_encode_blocks(blocks: torch.Tensor, pools: torch.Tensor,
                      proj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy RVQ of (N, B) blocks against ``pools`` (L, K, B).

    Returns (indices (L, N) int64, reconstruction (N, B) float32). The search
    runs in fp32: bf16 rounding can flip sign bits near hyperplane boundaries.
    """
    residual = blocks.float()
    pools = pools.to(device=blocks.device, dtype=torch.float32)
    recon = torch.zeros_like(residual)
    indices = []
    for level in range(pools.shape[0]):
        idx = nearest_index(residual, proj)
        sel = pools[level][idx]
        recon += sel
        residual = residual - sel
        indices.append(idx)
    return torch.stack(indices), recon


def rvq_decode_blocks(indices: torch.Tensor, pools: torch.Tensor) -> torch.Tensor:
    """Sum of selected codewords: indices (L, N) -> (N, B) float32."""
    pools = pools.to(device=indices.device, dtype=torch.float32)
    idx = indices.long()
    out = pools[0][idx[0]].clone()
    for level in range(1, idx.shape[0]):
        out += pools[level][idx[level]]
    return out


def encode(w: torch.Tensor, pools: torch.Tensor, proj: torch.Tensor
           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise a factor of any shape. Returns (indices (L, n_blocks), w_hat with w's shape)."""
    idx, rec = rvq_encode_blocks(blockify(w.detach(), pools.shape[-1]), pools, proj)
    return idx, unblockify(rec, w.shape)


def decode(indices: torch.Tensor, shape, pools: torch.Tensor) -> torch.Tensor:
    return unblockify(rvq_decode_blocks(indices, pools), shape)


def ste(w: torch.Tensor, w_hat: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward value ``w_hat``, identity gradient to ``w``."""
    return w_hat.detach() + (w - w.detach())
