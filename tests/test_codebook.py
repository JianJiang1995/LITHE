import hashlib

import numpy as np
import torch

from lithe import codebook as cb

# Pinned digests: any change to the construction must break these.
INT_TABLE_SHA256 = "eb1f4de0d64df9980436232299238e56ff04efa9ab075ded3f72c2a821ad0301"


def test_sylvester_definition():
    h = cb.sylvester_hadamard(64).astype(np.int64)
    assert (h @ h.T == 64 * np.eye(64, dtype=np.int64)).all()
    for t in range(64):
        for j in range(64):
            assert h[t, j] == (-1) ** bin(t & j).count("1")


def test_generator_is_first_nine_columns():
    u = cb.hadamard_generator().astype(np.int64)
    assert u.shape == (64, 9)
    assert (u.T @ u == 64 * np.eye(9, dtype=np.int64)).all()
    assert (u[:, 0] == 1).all()


def test_sign_table_bit_order():
    s = cb.sign_table()
    assert s.shape == (512, 9)
    assert (s[0] == -1).all() and (s[511] == 1).all()
    assert s[1, 0] == 1 and (s[1, 1:] == -1).all()       # bit 0 is the LSB
    assert s[256, 8] == 1 and (s[256, :8] == -1).all()


def test_codebook_hash_and_injectivity():
    table = cb.integer_codewords()
    assert table.shape == (512, 64) and table.dtype == np.int8
    assert cb.codebook_sha256() == INT_TABLE_SHA256
    assert hashlib.sha256(table.tobytes()).hexdigest() == INT_TABLE_SHA256
    assert len({row.tobytes() for row in table}) == 512
    assert set(np.unique(table).tolist()) <= {-9, -7, -5, -3, -1, 1, 3, 5, 7, 9}


def test_closed_form_matches_paper_formula():
    c = cb.codewords().double()
    u = torch.from_numpy(cb.hadamard_generator().astype(np.float64))
    s = torch.from_numpy(cb.sign_table().astype(np.float64))
    ref = s @ u.T / 3.0
    assert torch.equal(c, ref.float().double())
    assert abs(ref.pow(2).mean().item() - 1.0) < 1e-12    # unit per-coordinate variance
    assert torch.allclose(c.norm(dim=1), torch.full((512,), 8.0, dtype=torch.float64))


def test_twin_pool_scales():
    pa, pb = cb.twin_pools()
    c = cb.codewords()
    assert pa.shape == pb.shape == (3, 512, 64)
    for level, s in enumerate((0.06, 0.06, 0.06)):
        assert torch.allclose(pa[level], s * c, rtol=1e-6, atol=0)
    for level, s in enumerate((0.0025, 0.001, 0.0005)):
        assert torch.allclose(pb[level], s * c, rtol=1e-6, atol=0)
