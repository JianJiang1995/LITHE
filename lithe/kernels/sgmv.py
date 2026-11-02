"""Two-stage per-sample LoRA delta over decoded factors (SGMV-style, Punica lineage).

For a batch in which sample ``b`` uses its own factors ``A_b`` (r, d_in) and
``B_b`` (d_out, r) with scale ``s_b``::

    delta[b, s, o] = s_b * sum_k B_b[o, k] * sum_d A_b[k, d] * x[b, s, d]

Stage 1 (shrink): ``mid = s_b * (x_b @ A_b.T)``, grid ``(B, ceil(seq / BLOCK_S))``.
Stage 2 (expand): ``delta = mid @ B_b.T``, grid ``(B, ceil(seq / BLOCK_S), ceil(out / BLOCK_O))``.

A single fused (B, seq tile, out tile) kernel would redo the d_in reduction
once per output tile; lifting it into its own kernel and writing the small
``mid`` (B, seq, r) tensor is much faster on FLUX shapes. Storage is bf16 with
fp32 accumulation inside ``tl.dot``; ``mid`` is rounded to bf16 between the
stages. Ranks below 16 are zero-padded to 16 (``tl.dot`` minimum on Ampere).

This is the kernel of the paper's decoded same-batch path and of the
head-to-head against the vLLM release of the Punica / S-LoRA SGMV kernel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._triton import HAS_TRITON, tl, triton, use_triton

if HAS_TRITON:

    @triton.jit
    def _lora_shrink_kernel(
        X_ptr,           # [B, seq, in_dim] bf16
        SA_ptr,          # [B, r,   in_dim] bf16
        SCALE_ptr,       # [B]              bf16
        MID_ptr,         # [B, seq, r]      bf16
        x_b, x_s, x_d,
        a_b, a_r, a_d,
        m_b, m_s, m_r,
        SEQ: tl.constexpr,
        IN_D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)

        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_r = tl.arange(0, R)
        offs_k = tl.arange(0, BLOCK_K)
        mask_s = offs_s < SEQ

        acc = tl.zeros((BLOCK_S, R), dtype=tl.float32)
        for k_start in range(0, IN_D, BLOCK_K):
            ok = k_start + offs_k
            mk = ok < IN_D
            x_ptrs = X_ptr + pid_b * x_b + offs_s[:, None] * x_s + ok[None, :] * x_d
            x_tile = tl.load(x_ptrs, mask=mask_s[:, None] & mk[None, :], other=0.0)
            sa_ptrs = SA_ptr + pid_b * a_b + offs_r[:, None] * a_r + ok[None, :] * a_d
            sa_tile = tl.load(sa_ptrs, mask=mk[None, :], other=0.0)
            # [BLOCK_S, BLOCK_K] @ [BLOCK_K, R] -> [BLOCK_S, R]
            acc += tl.dot(x_tile, tl.trans(sa_tile))

        # the per-sample scale is applied on the small mid tensor, not on the delta
        scale = tl.load(SCALE_ptr + pid_b).to(tl.float32)
        acc = acc * scale
        m_ptrs = MID_ptr + pid_b * m_b + offs_s[:, None] * m_s + offs_r[None, :] * m_r
        tl.store(m_ptrs, acc.to(tl.bfloat16), mask=mask_s[:, None])

    @triton.jit
    def _lora_expand_kernel(
        MID_ptr,    # [B, seq, r]    bf16
        SB_ptr,     # [B, out, r]    bf16
        OUT_ptr,    # [B, seq, out]  bf16
        m_b, m_s, m_r,
        b_b, b_o, b_r,
        o_b, o_s, o_o,
        SEQ: tl.constexpr,
        OUT_D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_o = tl.program_id(2)

        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_r = tl.arange(0, R)
        mask_s = offs_s < SEQ
        mask_o = offs_o < OUT_D

        m_ptrs = MID_ptr + pid_b * m_b + offs_s[:, None] * m_s + offs_r[None, :] * m_r
        mid_tile = tl.load(m_ptrs, mask=mask_s[:, None], other=0.0)
        sb_ptrs = SB_ptr + pid_b * b_b + offs_o[:, None] * b_o + offs_r[None, :] * b_r
        sb_tile = tl.load(sb_ptrs, mask=mask_o[:, None], other=0.0)
        # [BLOCK_S, R] @ [R, BLOCK_O] -> [BLOCK_S, BLOCK_O]; K = r fits one tensor-core tile
        out = tl.dot(mid_tile, tl.trans(sb_tile))
        out_ptrs = OUT_ptr + pid_b * o_b + offs_s[:, None] * o_s + offs_o[None, :] * o_o
        tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_s[:, None] & mask_o[None, :])


def per_sample_lora_delta_reference(x: torch.Tensor, sA: torch.Tensor, sB: torch.Tensor,
                                    scaling: torch.Tensor) -> torch.Tensor:
    """bmm reference: x (B, seq, d_in), sA (B, r, d_in), sB (B, d_out, r), scaling (B,)."""
    bsz = x.shape[0]
    mid = torch.bmm(sA.to(x.dtype), x.transpose(1, 2))          # [B, r, seq]
    delta = torch.bmm(sB.to(x.dtype), mid).transpose(1, 2)      # [B, seq, out]
    return delta * scaling.to(x.dtype).view(bsz, 1, 1)


def per_sample_lora_delta(x: torch.Tensor, sA: torch.Tensor, sB: torch.Tensor,
                          scaling: torch.Tensor) -> torch.Tensor:
    """Per-sample LoRA delta ``[B, seq, d_out]``; Triton on CUDA bf16, else the bmm reference."""
    if x.dim() != 3:
        raise ValueError(f"x must be 3D, got {tuple(x.shape)}")
    bsz, seq, in_dim = x.shape
    ba, r, in_a = sA.shape
    bb, out_dim, rb = sB.shape
    if ba != bsz or bb != bsz or scaling.shape[0] != bsz:
        raise ValueError(f"batch mismatch x={bsz} sA={ba} sB={bb} scaling={scaling.shape[0]}")
    if in_a != in_dim or r != rb:
        raise ValueError("factor shapes do not match x")
    if not use_triton(x) or x.dtype != torch.bfloat16:
        return per_sample_lora_delta_reference(x, sA, sB, scaling)

    sA = sA.to(x.dtype)
    sB = sB.to(x.dtype)
    if r < 16:  # tl.dot needs K >= 16: zero-pad the rank (exact)
        sA = F.pad(sA, (0, 0, 0, 16 - r))
        sB = F.pad(sB, (0, 16 - r))
        r = 16
    x, sA, sB, scaling = x.contiguous(), sA.contiguous(), sB.contiguous(), scaling.contiguous()

    mid = torch.empty((bsz, seq, r), dtype=torch.bfloat16, device=x.device)
    block_s1, block_k = 64, 64
    if seq < 64:
        block_s1 = 32 if seq >= 32 else 16
    _lora_shrink_kernel[(bsz, triton.cdiv(seq, block_s1))](
        x, sA, scaling, mid,
        x.stride(0), x.stride(1), x.stride(2),
        sA.stride(0), sA.stride(1), sA.stride(2),
        mid.stride(0), mid.stride(1), mid.stride(2),
        SEQ=seq, IN_D=in_dim, R=r, BLOCK_S=block_s1, BLOCK_K=block_k,
        num_warps=4, num_stages=3,
    )

    out = torch.empty((bsz, seq, out_dim), dtype=torch.bfloat16, device=x.device)
    block_s2, block_o = 64, 64
    if seq < 64:
        block_s2 = 32 if seq >= 32 else 16
    _lora_expand_kernel[(bsz, triton.cdiv(seq, block_s2), triton.cdiv(out_dim, block_o))](
        mid, sB, out,
        mid.stride(0), mid.stride(1), mid.stride(2),
        sB.stride(0), sB.stride(1), sB.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        SEQ=seq, OUT_D=out_dim, R=r, BLOCK_S=block_s2, BLOCK_O=block_o,
        num_warps=4, num_stages=3,
    )
    return out
