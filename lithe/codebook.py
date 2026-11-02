"""Deterministic Hadamard-9bit twin codebook.

The codebook is a fixed function of the order-64 Sylvester-Hadamard matrix;
nothing is learned, seeded or stored. For index k in [0, 512):

    U        = H_syl[:, 0:9]                      (64 x 9, entries +-1)
    s_k[q]   = +1 if bit q of k is set else -1    (q = 0 is the LSB)
    c_k      = U s_k / sqrt(9)                    (64-vector, unit per-coordinate variance)

Side-specific RVQ pools scale the same codewords:

    pool_A[l] = SIGMA_A      * C   for l = 0, 1, 2
    pool_B[l] = SIGMA_B[l]   * C

All codewords have the same norm, so nearest-neighbour search against a pool
reduces to the sign of the projection onto the nine Hadamard columns
(see ``lithe.quant.nearest_index``).
"""
from __future__ import annotations

import hashlib
from typing import Sequence, Tuple

import numpy as np
import torch

BLOCK_SIZE = 64
N_BITS = 9
K = 1 << N_BITS
NUM_LEVELS = 3
SIGMA_A: Tuple[float, ...] = (0.06, 0.06, 0.06)
SIGMA_B: Tuple[float, ...] = (0.0025, 0.001, 0.0005)


def sylvester_hadamard(n: int) -> np.ndarray:
    """H[t, j] = (-1)^popcount(t & j), the order-n Sylvester-Hadamard matrix."""
    if n <= 0 or n & (n - 1):
        raise ValueError(f"n must be a power of two, got {n}")
    t = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    parity = np.vectorize(lambda v: bin(v).count("1") & 1)(t & j)
    return (1 - 2 * parity).astype(np.int8)


def hadamard_generator(block_size: int = BLOCK_SIZE, n_bits: int = N_BITS) -> np.ndarray:
    """U = first ``n_bits`` columns of H_syl, shape (block_size, n_bits), int8 +-1."""
    if block_size < n_bits:
        raise ValueError("block_size must be >= n_bits")
    return np.ascontiguousarray(sylvester_hadamard(block_size)[:, :n_bits])


def sign_table(n_bits: int = N_BITS) -> np.ndarray:
    """s_k for every k, shape (2**n_bits, n_bits), int8 +-1 (+1 where bit q of k is set)."""
    k = np.arange(1 << n_bits)[:, None]
    q = np.arange(n_bits)[None, :]
    return (2 * ((k >> q) & 1) - 1).astype(np.int8)


def integer_codewords(block_size: int = BLOCK_SIZE, n_bits: int = N_BITS) -> np.ndarray:
    """sqrt(n_bits) * c_k = U s_k as exact integers, shape (K, block_size), int8."""
    u = hadamard_generator(block_size, n_bits).astype(np.int32)
    s = sign_table(n_bits).astype(np.int32)
    return (s @ u.T).astype(np.int8)


def codewords(block_size: int = BLOCK_SIZE, n_bits: int = N_BITS) -> torch.Tensor:
    """Unit-variance codeword matrix C, shape (K, block_size), float32."""
    c = integer_codewords(block_size, n_bits).astype(np.float64) / np.sqrt(n_bits)
    return torch.from_numpy(c.astype(np.float32))


def build_pools(scales: Sequence[float], block_size: int = BLOCK_SIZE,
                n_bits: int = N_BITS) -> torch.Tensor:
    """Stack of scaled pools, shape (len(scales), K, block_size), float32."""
    c = integer_codewords(block_size, n_bits).astype(np.float64) / np.sqrt(n_bits)
    pools = np.stack([float(s) * c for s in scales]).astype(np.float32)
    return torch.from_numpy(pools)


def twin_pools(block_size: int = BLOCK_SIZE, n_bits: int = N_BITS,
               sigma_a: Sequence[float] = SIGMA_A,
               sigma_b: Sequence[float] = SIGMA_B) -> Tuple[torch.Tensor, torch.Tensor]:
    """(pool_A, pool_B), each (L, K, block_size) float32."""
    return build_pools(sigma_a, block_size, n_bits), build_pools(sigma_b, block_size, n_bits)


def projection_matrix(block_size: int = BLOCK_SIZE, n_bits: int = N_BITS) -> torch.Tensor:
    """U as float32 (block_size, n_bits); used for the closed-form search."""
    return torch.from_numpy(hadamard_generator(block_size, n_bits).astype(np.float32))


def codebook_sha256(block_size: int = BLOCK_SIZE, n_bits: int = N_BITS) -> str:
    """SHA-256 of the exact integer codeword table (int8, row-major)."""
    return hashlib.sha256(integer_codewords(block_size, n_bits).tobytes()).hexdigest()
