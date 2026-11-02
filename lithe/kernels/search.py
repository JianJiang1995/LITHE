"""Closed-form nearest-codeword search as one Triton kernel.

All codewords of a pool have the same norm, so the nearest codeword to a block
``t`` has bit ``q`` set exactly when ``<t, U[:, q]> > 0`` (``lithe.quant``).
One program handles ``BLOCK_N`` blocks: it loads the (BLOCK_N, 64) tile, forms
the nine projections, packs the sign bits into the index and gathers the
selected pool row, without ever materialising a K x 64 distance matrix.

The paper quotes this kernel as a per-block search microbenchmark (52 us vs.
135 us for a ``cdist`` baseline at N = 2048, bit-exact). The paper's adapters
were trained with the PyTorch search; ``rvq_encode_blocks_fast`` is offered as
a drop-in for ``lithe.quant.rvq_encode_blocks``. Indices agree with the PyTorch
closed form except when a projection is within fp32 rounding of zero, where
the summation order can flip the sign (``tests/test_kernels.py`` checks this).
"""
from __future__ import annotations

from typing import Tuple

import torch

from .. import quant
from ._triton import HAS_TRITON, tl, triton, use_triton

if HAS_TRITON:

    @triton.jit
    def _lattice_search_kernel(
        T_ptr,           # (N, B) fp32 blocks
        H_ptr,           # (n_bits, B) fp32 hyperplanes (+-1), i.e. U.T
        Pool_ptr,        # (K, B) fp32 codebook level
        Idx_ptr,         # (N,) int64 output indices
        Recon_ptr,       # (N, B) fp32 selected codewords
        N: tl.constexpr,
        B: tl.constexpr,
        NBITS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_b = tl.arange(0, B)
        t = tl.load(T_ptr + offs_n[:, None] * B + offs_b[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)

        idx_acc = tl.zeros((BLOCK_N,), dtype=tl.int64)
        for i in tl.static_range(0, NBITS):
            h = tl.load(H_ptr + i * B + offs_b).to(tl.float32)
            ip = tl.sum(t * h[None, :], axis=1)
            bit = (ip > 0.0).to(tl.int64)
            idx_acc = idx_acc + (bit << i)

        rec = tl.load(Pool_ptr + idx_acc[:, None] * B + offs_b[None, :], mask=mask_n[:, None], other=0.0)
        tl.store(Idx_ptr + offs_n, idx_acc, mask=mask_n)
        tl.store(Recon_ptr + offs_n[:, None] * B + offs_b[None, :], rec, mask=mask_n[:, None])


def lattice_search_reference(blocks: torch.Tensor, proj: torch.Tensor,
                             pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch closed form: (indices (N,) int64, selected codewords (N, B) fp32)."""
    idx = quant.nearest_index(blocks, proj)
    return idx, pool.to(device=blocks.device, dtype=torch.float32)[idx]


def lattice_search(blocks: torch.Tensor, proj: torch.Tensor,
                   pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest codeword of every block. blocks (N, B); proj = U (B, n_bits); pool (K, B)."""
    n, bs = blocks.shape
    n_bits = proj.shape[1]
    if proj.shape[0] != bs or pool.shape != (1 << n_bits, bs):
        raise ValueError("shape mismatch between blocks, proj and pool")
    if not use_triton(blocks):
        return lattice_search_reference(blocks, proj, pool)
    t = blocks.detach().contiguous().to(torch.float32)
    h = proj.t().contiguous().to(t.device, dtype=torch.float32)
    p = pool.contiguous().to(t.device, dtype=torch.float32)
    indices = torch.empty(n, dtype=torch.int64, device=t.device)
    recon = torch.empty(n, bs, dtype=torch.float32, device=t.device)
    block_n = 64
    _lattice_search_kernel[(triton.cdiv(n, block_n),)](
        t, h, p, indices, recon, N=n, B=bs, NBITS=n_bits, BLOCK_N=block_n, num_warps=4,
    )
    return indices, recon


@torch.no_grad()
def rvq_encode_blocks_fast(blocks: torch.Tensor, pools: torch.Tensor,
                           proj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy twin RVQ like ``lithe.quant.rvq_encode_blocks``, one kernel launch per level."""
    residual = blocks.float()
    pools = pools.to(device=blocks.device, dtype=torch.float32)
    recon = torch.zeros_like(residual)
    indices = []
    for level in range(pools.shape[0]):
        idx, sel = lattice_search(residual, proj, pools[level])
        recon += sel
        residual = residual - sel
        indices.append(idx)
    return torch.stack(indices), recon
