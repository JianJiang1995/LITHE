import torch

from lithe import codebook as cb
from lithe import quant


def _cdist_rvq(x, pools):
    residual, idx, rec = x.clone(), [], torch.zeros_like(x)
    for level in range(pools.shape[0]):
        j = torch.cdist(residual, pools[level]).argmin(dim=1)
        idx.append(j)
        rec += pools[level][j]
        residual = residual - pools[level][j]
    return torch.stack(idx), rec


def test_closed_form_equals_cdist():
    torch.manual_seed(0)
    pa, pb = cb.twin_pools()
    proj = cb.projection_matrix()
    for pools, scale in ((pa, 0.08), (pb, 0.003)):
        x = scale * torch.randn(8192, 64)
        idx, rec = quant.rvq_encode_blocks(x, pools, proj)
        ref_idx, ref_rec = _cdist_rvq(x, pools)
        assert torch.equal(idx, ref_idx)
        assert torch.allclose(rec, ref_rec, atol=1e-7)


def test_decode_of_encode_is_reconstruction():
    torch.manual_seed(1)
    pa, _ = cb.twin_pools()
    proj = cb.projection_matrix()
    w = 0.05 * torch.randn(16, 3072)
    idx, w_hat = quant.encode(w, pa, proj)
    assert idx.shape == (3, 16 * 3072 // 64)
    assert torch.equal(quant.decode(idx, w.shape, pa), w_hat)


def test_roundtrip_error_structure():
    """The nine U columns are orthogonal, so RVQ acts per Hadamard coordinate.

    The component of a block outside col(U) is never touched, and each of the
    nine coordinates follows the scalar greedy recursion e_l = e_{l-1} - sign(e_{l-1}) d_l
    with step d_l = 8 sigma_l / 3. This gives the exact reconstruction error.
    """
    torch.manual_seed(2)
    pa, pb = cb.twin_pools()
    proj = cb.projection_matrix()
    u = proj / 8.0                                      # orthonormal basis of col(U)
    for pools, sigmas, scale in ((pa, cb.SIGMA_A, 0.06), (pb, cb.SIGMA_B, 0.0025)):
        x = scale * torch.randn(4096, 64, dtype=torch.float64)
        _, rec = quant.rvq_encode_blocks(x.float(), pools, proj)
        err = x - rec.double()
        perp = x - x @ u.double() @ u.double().T
        assert torch.allclose(err - err @ u.double() @ u.double().T, perp, atol=1e-6)
        e = x @ u.double()                              # (N, 9) coordinates
        for s in sigmas:
            e = e - torch.where(e > 0, 1.0, -1.0) * (8.0 * s / 3.0)
        assert torch.allclose(err @ u.double(), e, atol=1e-6)
        if pools is pb:                                  # decaying B scales: bounded per-coordinate error
            t = x @ u.double()
            inside = t.abs() <= 8.0 * sum(sigmas) / 3.0
            assert (e.abs()[inside] <= 8.0 * sigmas[0] / 3.0 + 1e-9).all()


def test_exact_lattice_points_roundtrip():
    torch.manual_seed(3)
    _, pb = cb.twin_pools()
    proj = cb.projection_matrix()
    true_idx = torch.randint(0, 512, (3, 2048))
    x = quant.rvq_decode_blocks(true_idx, pb)
    idx, rec = quant.rvq_encode_blocks(x, pb, proj)
    assert torch.equal(idx, true_idx)
    assert torch.equal(rec, x)


def test_padding_and_shapes():
    pa, _ = cb.twin_pools()
    proj = cb.projection_matrix()
    w = torch.randn(5, 37) * 0.05                      # 185 values -> 3 blocks, padded
    idx, w_hat = quant.encode(w, pa, proj)
    assert idx.shape == (3, 3) and w_hat.shape == w.shape


def test_ste_gradient_is_identity():
    w = torch.randn(4, 64, requires_grad=True)
    w_hat = torch.randn(4, 64)
    out = quant.ste(w, w_hat)
    assert torch.equal(out.detach(), w_hat)
    g = torch.randn(4, 64)
    out.backward(g)
    assert torch.equal(w.grad, g)
