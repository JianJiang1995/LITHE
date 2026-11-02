"""Bit-packed GPU residency and the fused whole-adapter decode.

Residency form (paper Sec. 4.3 and App. "N_max co-resident inventory"): the
index stream of one adapter is laid out (block, level)-major, every index is
cut to 9 bits (K = 512) and 7 indices share one uint64 word (63 of 64 bits
used, the top bit is always 0). Index ``p`` of the flat stream lives in word
``p // 7`` at bits ``[9 * (p % 7), 9 * (p % 7) + 9)``. For the Kohya-304
FLUX.1-dev layout an adapter holds 3,589,632 indices, i.e. 512,805 words =
4.10 MB.

Decode: one Triton program per 64-value block reads its L indices from the
packed words, gathers the selected rows of the fp32 pool of its side (A or B)
and writes the fp32 sum as bf16. All blocks of all layers of one adapter are
decoded by a single launch; ``split_decoded`` then returns per-module views.
This is the kernel timed at 4.56 ms (warm mean over 200 calls, A100) in the
paper. The output equals ``lithe.quant.decode(...)`` cast to bf16 bit for bit
(same fp32 summation order).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from .. import quant
from ._triton import HAS_TRITON, tl, triton, use_triton

INDEX_BITS = 9
PER_WORD = 7          # 7 x 9 = 63 bits per uint64 word
INDEX_MASK = (1 << INDEX_BITS) - 1


# ---------------------------------------------------------------------------
# Packing (host side)
# ---------------------------------------------------------------------------
def pack_indices(indices) -> torch.Tensor:
    """Pack a flat sequence of 9-bit indices into a uint64 tensor (CPU), 7 per word."""
    flat = np.asarray(indices).reshape(-1).astype(np.uint64)
    if flat.size and int(flat.max()) > INDEX_MASK:
        raise ValueError("indices must be < 512 for 9-bit packing")
    n = flat.size
    n_words = (n + PER_WORD - 1) // PER_WORD
    pad = n_words * PER_WORD - n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint64)])
    flat = flat.reshape(n_words, PER_WORD)
    packed = np.zeros(n_words, dtype=np.uint64)
    for i in range(PER_WORD):
        packed |= flat[:, i] << np.uint64(i * INDEX_BITS)
    return torch.from_numpy(packed.astype(np.int64)).view(torch.uint64)


def unpack_indices(words: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of ``pack_indices`` with torch ops on ``words.device``; returns (n,) int64."""
    w = words.view(torch.int64)            # bit 63 is never set, so the shift is logical
    shifts = torch.arange(PER_WORD, device=w.device, dtype=torch.int64) * INDEX_BITS
    out = (w[:, None] >> shifts[None, :]) & INDEX_MASK
    return out.reshape(-1)[:n]


@dataclass
class PackedLayout:
    """Where every module's A and B blocks sit in the packed stream of one adapter.

    Adapters that cover the same modules with the same shapes share one layout,
    so a resident bank needs one layout plus one packed tensor per slot.
    """

    names: List[str]
    shapes_a: List[Tuple[int, int]]
    shapes_b: List[Tuple[int, int]]
    a_off: List[int]
    b_off: List[int]
    sides: torch.Tensor                     # (total_blocks,) int8, 0 = A block, 1 = B block
    num_levels: int
    block_size: int
    scaling: float                          # alpha / rank
    _sides_dev: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    @property
    def total_blocks(self) -> int:
        return int(self.sides.numel())

    @property
    def num_indices(self) -> int:
        return self.total_blocks * self.num_levels

    @property
    def num_words(self) -> int:
        return (self.num_indices + PER_WORD - 1) // PER_WORD

    def sides_on(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._sides_dev:
            self._sides_dev[key] = self.sides.to(device).contiguous()
        return self._sides_dev[key]

    @classmethod
    def from_adapter(cls, ad) -> "PackedLayout":
        names, sa, sb, a_off, b_off, sides = [], [], [], [], [], []
        cursor = 0
        for m in ad.modules:
            n_a = quant.num_blocks(m["shape_a"], ad.block_size)
            n_b = quant.num_blocks(m["shape_b"], ad.block_size)
            names.append(m["name"])
            sa.append(tuple(int(s) for s in m["shape_a"]))
            sb.append(tuple(int(s) for s in m["shape_b"]))
            a_off.append(cursor)
            b_off.append(cursor + n_a)
            sides.append(torch.zeros(n_a, dtype=torch.int8))
            sides.append(torch.ones(n_b, dtype=torch.int8))
            cursor += n_a + n_b
        return cls(names=names, shapes_a=sa, shapes_b=sb, a_off=a_off, b_off=b_off,
                   sides=torch.cat(sides).contiguous(), num_levels=ad.num_levels,
                   block_size=ad.block_size, scaling=float(ad.alpha) / ad.rank)

    def matches(self, ad) -> bool:
        return (len(ad.modules) == len(self.names)
                and all(m["name"] == n and tuple(m["shape_a"]) == a and tuple(m["shape_b"]) == b
                        for m, n, a, b in zip(ad.modules, self.names, self.shapes_a, self.shapes_b)))


def flat_indices(ad, layout: PackedLayout) -> np.ndarray:
    """The adapter's indices in packed-stream order: per module A blocks then B blocks,
    each block contributing its L indices (block, level)-major."""
    parts = []
    for name in layout.names:
        idx = ad.indices[name]
        parts.append(np.ascontiguousarray(idx["A"].T).reshape(-1))
        parts.append(np.ascontiguousarray(idx["B"].T).reshape(-1))
    return np.concatenate(parts).astype(np.uint16)


def pack_adapter(ad, layout: PackedLayout = None) -> Tuple[torch.Tensor, PackedLayout]:
    """(packed uint64 words on CPU, layout) for one ``LitheAdapter``."""
    if ad.n_bits != INDEX_BITS:
        raise ValueError(f"packing assumes 9-bit indices, adapter has n_bits={ad.n_bits}")
    layout = layout or PackedLayout.from_adapter(ad)
    if not layout.matches(ad):
        raise ValueError("adapter modules do not match the layout")
    return pack_indices(flat_indices(ad, layout)), layout


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _decode_packed_kernel(
        packed_ptr,        # uint64 words of one adapter
        pool_a_ptr,        # fp32 (L, K, B)
        pool_b_ptr,        # fp32 (L, K, B)
        sides_ptr,         # int8 (total_blocks,), 0 = A, 1 = B
        out_ptr,           # bf16 (total_blocks, B)
        L: tl.constexpr,
        K: tl.constexpr,
        B: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        b_offsets = tl.arange(0, B)
        accum = tl.zeros((B,), dtype=tl.float32)
        side = tl.load(sides_ptr + block_id)
        base_pool = tl.where(side == 0, pool_a_ptr, pool_b_ptr)
        # index of (block_id, level l) is flat position block_id * L + l:
        # word (pos // 7), bits [(pos % 7) * 9, (pos % 7) * 9 + 9)
        for l in range(L):
            logical_pos = block_id * L + l
            group_idx = logical_pos // 7
            bit_in_group = (logical_pos % 7) * 9
            word = tl.load(packed_ptr + group_idx)
            idx = ((word >> bit_in_group) & 0x1FF).to(tl.int32)
            ptrs = base_pool + l * K * B + idx * B + b_offsets
            accum += tl.load(ptrs)
        tl.store(out_ptr + block_id * B + b_offsets, accum.to(tl.bfloat16))

    @triton.jit
    def _decode_int16_kernel(
        idx_ptr,           # int16 (total_blocks, L), (block, level)-major
        sides_ptr,         # int32 (total_blocks,)
        pool_a_ptr, pool_b_ptr, out_ptr,
        L: tl.constexpr, K: tl.constexpr, B: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        b_offsets = tl.arange(0, B)
        accum = tl.zeros((B,), dtype=tl.float32)
        side = tl.load(sides_ptr + block_id)
        base_pool = tl.where(side == 0, pool_a_ptr, pool_b_ptr)
        for l in range(L):
            idx = tl.load(idx_ptr + block_id * L + l).to(tl.int32)
            ptrs = base_pool + l * K * B + idx * B + b_offsets
            accum += tl.load(ptrs)
        tl.store(out_ptr + block_id * B + b_offsets, accum.to(tl.bfloat16))


def _check_pools(pool_a: torch.Tensor, pool_b: torch.Tensor, layout: PackedLayout):
    if pool_a.shape != pool_b.shape or pool_a.dim() != 3:
        raise ValueError("pools must both be (L, K, B)")
    if pool_a.shape[0] != layout.num_levels or pool_a.shape[2] != layout.block_size:
        raise ValueError("pool shape does not match the layout")


def decode_packed(words: torch.Tensor, layout: PackedLayout, pool_a: torch.Tensor,
                  pool_b: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Decode every block of one packed adapter -> (total_blocks, block_size) bf16.

    ``pool_a`` / ``pool_b`` are the fp32 (L, K, B) pools (``lithe.codebook.build_pools``)
    on the same device as ``words``. Uses the Triton kernel on CUDA, otherwise the
    PyTorch reference.
    """
    _check_pools(pool_a, pool_b, layout)
    if not (use_triton(words) and dtype == torch.bfloat16):
        return decode_packed_reference(words, layout, pool_a, pool_b, dtype)
    L, K, B = pool_a.shape
    out = torch.empty((layout.total_blocks, B), dtype=dtype, device=words.device)
    _decode_packed_kernel[(layout.total_blocks,)](
        words, pool_a.float().contiguous(), pool_b.float().contiguous(),
        layout.sides_on(words.device), out, L=L, K=K, B=B,
    )
    return out


def decode_int16(idx: torch.Tensor, layout: PackedLayout, pool_a: torch.Tensor,
                 pool_b: torch.Tensor) -> torch.Tensor:
    """Same decode from an unpacked (total_blocks, L) int16 index tensor (residency A/B check)."""
    _check_pools(pool_a, pool_b, layout)
    if not use_triton(idx):
        flat = idx.reshape(-1).long()
        return _decode_flat_reference(flat, layout, pool_a, pool_b, torch.bfloat16)
    L, K, B = pool_a.shape
    out = torch.empty((layout.total_blocks, B), dtype=torch.bfloat16, device=idx.device)
    _decode_int16_kernel[(layout.total_blocks,)](
        idx.contiguous(), layout.sides_on(idx.device).to(torch.int32),
        pool_a.float().contiguous(), pool_b.float().contiguous(), out, L=L, K=K, B=B,
    )
    return out


def _decode_flat_reference(flat: torch.Tensor, layout: PackedLayout, pool_a, pool_b, dtype):
    L = layout.num_levels
    idx = flat.view(layout.total_blocks, L)
    pool_a = pool_a.to(device=flat.device, dtype=torch.float32)
    pool_b = pool_b.to(device=flat.device, dtype=torch.float32)
    is_b = layout.sides_on(flat.device).bool()
    out = torch.zeros(layout.total_blocks, pool_a.shape[-1], dtype=torch.float32, device=flat.device)
    for l in range(L):   # same fp32 summation order as the kernel
        out += torch.where(is_b[:, None], pool_b[l][idx[:, l]], pool_a[l][idx[:, l]])
    return out.to(dtype)


def decode_packed_reference(words: torch.Tensor, layout: PackedLayout, pool_a: torch.Tensor,
                            pool_b: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """PyTorch reference of ``decode_packed`` (any device)."""
    flat = unpack_indices(words, layout.num_indices)
    return _decode_flat_reference(flat, layout, pool_a, pool_b, dtype)


def split_decoded(out: torch.Tensor, layout: PackedLayout) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """name -> (A (r, d_in), B (d_out, r)) as views into the decode output."""
    flat = out.reshape(-1)
    bs = layout.block_size
    res = {}
    for name, sa, sb, ao, bo in zip(layout.names, layout.shapes_a, layout.shapes_b,
                                    layout.a_off, layout.b_off):
        na, nb = sa[0] * sa[1], sb[0] * sb[1]
        res[name] = (flat[ao * bs: ao * bs + na].view(*sa), flat[bo * bs: bo * bs + nb].view(*sb))
    return res
