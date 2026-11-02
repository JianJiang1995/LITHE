"""Index-resident per-sample LoRA delta with the decode fused into the matmul tiles.

Each wrapped linear keeps, for every resident adapter, only its index tensors
``idx_A`` (n_slots, L, n_blocks_A) and ``idx_B`` (n_slots, L, n_blocks_B), plus
ONE pair of codebook pools shared by all adapters and all samples. A batch is
routed by ``assign`` (B,): sample ``b`` uses adapter ``j = assign[b]``::

    A_j[r, d]   = sum_l pool_A[l, idx_A[j, l, blk(r, d)], pos(r, d)]
    B_j[o, r]   = sum_l pool_B[l, idx_B[j, l, blk(o, r)], pos(o, r)]
    delta[b]    = scaling * (x_b @ A_j.T) @ B_j.T

Kernel A decodes a (BR, 64) tile of ``A_j`` in registers for every 64-wide
slice of ``d_in`` and accumulates ``x_b @ A_j.T``; kernel B decodes a (BO, r)
tile of ``B_j`` and multiplies. Two launches per layer cover the whole batch;
the pools stay hot in L2 across all programs ("codebook-shared decode cache"
in the paper). Blocks follow the row-major flattening of ``lithe.quant``, so
``d_in`` and ``d_out * r`` must be multiples of the block size (true for all
FLUX.1-dev linears).

This is the LITHE kernel behind the decoded-compiled column of the paper's
wall-time table (with ``torch.compile`` applied to the transformer) and the
"shared cache" rows of the serving-kernel ablation. The pools are cast to the
activation dtype (bf16) before the gather, as in the original runs, so the
decoded factors match an fp32 decode to bf16 precision, not bit for bit.
"""
from __future__ import annotations

from typing import Optional

import torch

from ._triton import HAS_TRITON, tl, triton, use_triton

if HAS_TRITON:

    @triton.jit
    def _shared_decode_A_matmul_kernel(
        x_ptr, idx_pool_ptr, pools_A_ptr, assign_ptr, y_ptr,
        M, D_in, R,
        L: tl.constexpr, K_CB: tl.constexpr, B_CB: tl.constexpr,
        n_blocks_A,
        stride_x_b, stride_x_m, stride_x_d,
        stride_idx_n, stride_idx_l, stride_idx_b,
        stride_pA_l, stride_pA_k, stride_pA_b,
        stride_y_b, stride_y_m, stride_y_r,
        BM: tl.constexpr, BR: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_r = tl.program_id(2)

        # adapter slot of this sample
        lora_id = tl.load(assign_ptr + pid_b)
        lora_id = lora_id.to(tl.int32)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_r = pid_r * BR + tl.arange(0, BR)
        offs_b = tl.arange(0, B_CB)
        mask_m = offs_m < M
        mask_r = offs_r < R

        acc = tl.zeros((BM, BR), dtype=tl.float32)
        n_k_tiles = D_in // B_CB
        blocks_per_row = D_in // B_CB

        x_base = x_ptr + pid_b * stride_x_b
        idx_base = idx_pool_ptr + lora_id * stride_idx_n
        y_base = y_ptr + pid_b * stride_y_b

        for k_tile in range(0, n_k_tiles):
            d_start = k_tile * B_CB
            x_offs = offs_m[:, None] * stride_x_m + (d_start + offs_b)[None, :] * stride_x_d
            x_blk = tl.load(
                x_base + x_offs,
                mask=mask_m[:, None] & ((d_start + offs_b)[None, :] < D_in),
                other=0.0,
            )
            # row r of A covers blocks r * blocks_per_row ... ; this tile is block k_tile of each row
            block_ids = offs_r * blocks_per_row + k_tile
            sA_blk = tl.zeros((BR, B_CB), dtype=tl.float32)
            for l in tl.static_range(L):
                idx_offs = l * stride_idx_l + block_ids * stride_idx_b
                cb_idx = tl.load(idx_base + idx_offs, mask=mask_r, other=0)
                cb_idx = cb_idx.to(tl.int32)
                p_offs = l * stride_pA_l + cb_idx[:, None] * stride_pA_k + offs_b[None, :] * stride_pA_b
                p_val = tl.load(pools_A_ptr + p_offs, mask=mask_r[:, None], other=0.0)
                sA_blk = sA_blk + p_val.to(tl.float32)
            acc = acc + tl.dot(x_blk, tl.trans(sA_blk.to(x_blk.dtype)), out_dtype=tl.float32)

        y_offs = offs_m[:, None] * stride_y_m + offs_r[None, :] * stride_y_r
        if OUT_DTYPE == 0:
            tl.store(y_base + y_offs, acc.to(tl.float32), mask=mask_m[:, None] & mask_r[None, :])
        else:
            tl.store(y_base + y_offs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_r[None, :])

    @triton.jit
    def _shared_decode_B_matmul_kernel(
        y_lo_ptr, idx_pool_ptr, pools_B_ptr, assign_ptr, delta_ptr,
        M, D_out, R,
        L: tl.constexpr, K_CB: tl.constexpr, B_CB: tl.constexpr,
        n_blocks_B,
        stride_y_b, stride_y_m, stride_y_r,
        stride_idx_n, stride_idx_l, stride_idx_b,
        stride_pB_l, stride_pB_k, stride_pB_b,
        stride_d_b, stride_d_m, stride_d_o,
        BM: tl.constexpr, BO: tl.constexpr, BR: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_o = tl.program_id(2)

        lora_id = tl.load(assign_ptr + pid_b)
        lora_id = lora_id.to(tl.int32)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_o = pid_o * BO + tl.arange(0, BO)
        offs_r = tl.arange(0, BR)
        mask_m = offs_m < M
        mask_o = offs_o < D_out

        y_base = y_lo_ptr + pid_b * stride_y_b
        idx_base = idx_pool_ptr + lora_id * stride_idx_n
        d_base = delta_ptr + pid_b * stride_d_b

        y_offs = offs_m[:, None] * stride_y_m + offs_r[None, :] * stride_y_r
        y_blk = tl.load(y_base + y_offs, mask=mask_m[:, None] & (offs_r[None, :] < R), other=0.0)

        # B is flattened row-major: element (o, r) is position o * R + r
        flat_idx = offs_o[:, None] * R + offs_r[None, :]
        block_ids = flat_idx // B_CB
        pos_ids = flat_idx % B_CB

        sB_blk = tl.zeros((BO, BR), dtype=tl.float32)
        for l in tl.static_range(L):
            idx_offs = l * stride_idx_l + block_ids * stride_idx_b
            cb_idx = tl.load(
                idx_base + idx_offs,
                mask=mask_o[:, None] & (offs_r[None, :] < R),
                other=0,
            )
            cb_idx = cb_idx.to(tl.int32)
            p_offs = l * stride_pB_l + cb_idx * stride_pB_k + pos_ids * stride_pB_b
            p_val = tl.load(
                pools_B_ptr + p_offs,
                mask=mask_o[:, None] & (offs_r[None, :] < R),
                other=0.0,
            )
            sB_blk = sB_blk + p_val.to(tl.float32)

        out = tl.dot(y_blk, tl.trans(sB_blk.to(y_blk.dtype)), out_dtype=tl.float32)
        d_offs = offs_m[:, None] * stride_d_m + offs_o[None, :] * stride_d_o
        if OUT_DTYPE == 0:
            tl.store(d_base + d_offs, out.to(tl.float32), mask=mask_m[:, None] & mask_o[None, :])
        else:
            tl.store(d_base + d_offs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_o[None, :])


def _check(x, idx_A, pools_A, idx_B, pools_B, assign, R, D_out):
    if x.dim() != 3:
        raise ValueError(f"x must be (B, M, D_in), got {tuple(x.shape)}")
    bsz, _, d_in = x.shape
    L, _, bs = pools_A.shape
    if pools_B.shape != pools_A.shape:
        raise ValueError("pools_A and pools_B must have the same (L, K, B) shape")
    if idx_A.dim() != 3 or idx_B.dim() != 3 or idx_A.shape[1] != L or idx_B.shape[1] != L:
        raise ValueError("index pools must be (n_slots, L, n_blocks)")
    if R * d_in != idx_A.shape[2] * bs or D_out * R != idx_B.shape[2] * bs:
        raise ValueError("index pools do not match (R, D_in, D_out)")
    if d_in % bs or (D_out * R) % bs:
        raise ValueError("d_in and d_out * r must be multiples of the block size")
    if assign.shape != (bsz,):
        raise ValueError(f"assign must be ({bsz},)")


def shared_codebook_lora_delta_reference(x, idx_A, pools_A, idx_B, pools_B, assign, R, D_out,
                                         scaling: float, out_dtype: Optional[torch.dtype] = None):
    """PyTorch reference of ``shared_codebook_lora_delta`` with the same dtype steps."""
    _check(x, idx_A, pools_A, idx_B, pools_B, assign, R, D_out)
    out_dtype = out_dtype or x.dtype
    bsz, _, d_in = x.shape
    pa = pools_A.to(device=x.device, dtype=x.dtype).float()
    pb = pools_B.to(device=x.device, dtype=x.dtype).float()
    out = []
    for b in range(bsz):
        j = int(assign[b])
        ia, ib = idx_A[j].long(), idx_B[j].long()
        a = sum(pa[l][ia[l]] for l in range(pa.shape[0])).reshape(R, d_in).to(x.dtype)
        bm = sum(pb[l][ib[l]] for l in range(pb.shape[0])).reshape(D_out, R).to(x.dtype)
        y = (x[b].float() @ a.float().T).to(x.dtype)
        out.append((y.float() @ bm.float().T).to(out_dtype))
    delta = torch.stack(out)
    return delta * scaling if scaling != 1.0 else delta


def shared_codebook_lora_delta(x: torch.Tensor, idx_A: torch.Tensor, pools_A: torch.Tensor,
                               idx_B: torch.Tensor, pools_B: torch.Tensor, assign: torch.Tensor,
                               R: int, D_out: int, scaling: float,
                               out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Per-sample LoRA delta ``(B, M, D_out)`` decoded on the fly from index pools.

    x (B, M, D_in); idx_A (n_slots, L, n_blocks_A) and idx_B (n_slots, L, n_blocks_B)
    integer; pools (L, K, block) float; assign (B,) adapter slot per sample.
    """
    if not use_triton(x, idx_A, idx_B, pools_A, pools_B):
        return shared_codebook_lora_delta_reference(x, idx_A, pools_A, idx_B, pools_B, assign,
                                                    R, D_out, scaling, out_dtype)
    _check(x, idx_A, pools_A, idx_B, pools_B, assign, R, D_out)
    if out_dtype is None:
        out_dtype = x.dtype
    B, M, D_in = x.shape
    L, K_CB, B_CB = pools_A.shape
    n_blocks_A, n_blocks_B = idx_A.shape[2], idx_B.shape[2]
    if pools_A.dtype != x.dtype:
        pools_A = pools_A.to(x.dtype)
    if pools_B.dtype != x.dtype:
        pools_B = pools_B.to(x.dtype)

    idx_A_i32 = idx_A.to(torch.int32).contiguous()
    idx_B_i32 = idx_B.to(torch.int32).contiguous()
    pools_A_c = pools_A.contiguous()
    pools_B_c = pools_B.contiguous()
    x_c = x.contiguous()
    assign_c = assign.to(device=x.device, dtype=torch.int32).contiguous()

    BR_pad = max(16, 1 << (R - 1).bit_length())
    BM = 64
    y_lo = torch.empty((B, M, R), device=x.device, dtype=x.dtype)
    grid1 = (B, triton.cdiv(M, BM), triton.cdiv(R, BR_pad))
    _shared_decode_A_matmul_kernel[grid1](
        x_c, idx_A_i32, pools_A_c, assign_c, y_lo,
        M, D_in, R,
        L, K_CB, B_CB,
        n_blocks_A,
        x_c.stride(0), x_c.stride(1), x_c.stride(2),
        idx_A_i32.stride(0), idx_A_i32.stride(1), idx_A_i32.stride(2),
        pools_A_c.stride(0), pools_A_c.stride(1), pools_A_c.stride(2),
        y_lo.stride(0), y_lo.stride(1), y_lo.stride(2),
        BM=BM, BR=BR_pad,
        OUT_DTYPE=0 if x.dtype == torch.float32 else 1,
        num_warps=4,
    )

    BO = 64
    delta = torch.empty((B, M, D_out), device=x.device, dtype=out_dtype)
    grid2 = (B, triton.cdiv(M, BM), triton.cdiv(D_out, BO))
    _shared_decode_B_matmul_kernel[grid2](
        y_lo, idx_B_i32, pools_B_c, assign_c, delta,
        M, D_out, R,
        L, K_CB, B_CB,
        n_blocks_B,
        y_lo.stride(0), y_lo.stride(1), y_lo.stride(2),
        idx_B_i32.stride(0), idx_B_i32.stride(1), idx_B_i32.stride(2),
        pools_B_c.stride(0), pools_B_c.stride(1), pools_B_c.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        BM=BM, BO=BO, BR=BR_pad,
        OUT_DTYPE=0 if out_dtype == torch.float32 else 1,
        num_warps=4,
    )
    if scaling != 1.0:
        delta = delta * scaling
    return delta
