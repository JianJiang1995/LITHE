"""Kernels vs PyTorch references. CPU tests always run; GPU tests need CUDA + Triton."""
import numpy as np
import pytest
import torch

from lithe import codebook as cb
from lithe import kohya, quant
from lithe.fileio import LitheAdapter
from lithe.kernels import (HAS_TRITON, IndexLoRALinear, PackedLayout, PerSampleLoRALinear,
                           decode_packed, decode_packed_reference, lattice_search,
                           lattice_search_reference, pack_adapter, pack_indices,
                           per_sample_lora_delta, per_sample_lora_delta_reference,
                           rvq_encode_blocks_fast, shared_codebook_lora_delta,
                           shared_codebook_lora_delta_reference, split_decoded, unpack_indices)
from lithe.kernels.packed import decode_int16, flat_indices

gpu = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON), reason="needs CUDA and Triton")


def random_adapter(modules, rank, seed=0, key_format="diffusers"):
    rng = np.random.default_rng(seed)
    mods, idx = [], {}
    for name, d_in, d_out in modules:
        sa, sb = [rank, d_in], [d_out, rank]
        mods.append({"name": name, "shape_a": sa, "shape_b": sb})
        idx[name] = {"A": rng.integers(0, cb.K, (cb.NUM_LEVELS, quant.num_blocks(sa))).astype(np.uint16),
                     "B": rng.integers(0, cb.K, (cb.NUM_LEVELS, quant.num_blocks(sb))).astype(np.uint16)}
    return LitheAdapter(modules=mods, indices=idx, rank=rank, alpha=float(rank), key_format=key_format)


def ref_factors(ad, device="cpu"):
    pa, pb = (p.to(device) for p in cb.twin_pools())
    out = {}
    for m in ad.modules:
        i = ad.indices[m["name"]]
        a = quant.decode(torch.from_numpy(i["A"].astype(np.int64)).to(device), m["shape_a"], pa)
        b = quant.decode(torch.from_numpy(i["B"].astype(np.int64)).to(device), m["shape_b"], pb)
        out[m["name"]] = (a, b)
    return out


SMALL = [("blk.0.q", 128, 192), ("blk.0.mlp", 128, 320), ("blk.0.out", 320, 128), ("odd", 96, 40)]


# ----------------------------------------------------------------------------- CPU
def test_pack_roundtrip():
    rng = np.random.default_rng(1)
    for n in (1, 6, 7, 8, 700, 1001):
        v = rng.integers(0, 512, n)
        words = pack_indices(v)
        assert words.dtype == torch.uint64 and words.numel() == -(-n // 7)
        assert np.array_equal(unpack_indices(words, n).numpy(), v)
    with pytest.raises(ValueError):
        pack_indices([512])


def test_packed_layout_size_matches_kohya_304():
    # Kohya-304 FLUX.1-dev shapes: 3,589,632 indices -> 512,805 words = 4.10 MB
    shapes = ([(3072, 9216)] * 76 + [(3072, 18432)] * 38 + [(3072, 3072)] * 38 + [(3072, 12288)] * 38
              + [(12288, 3072)] * 38 + [(3072, 21504)] * 38 + [(15360, 3072)] * 38)
    mods = [{"name": f"m{i}", "shape_a": [16, a], "shape_b": [b, 16]} for i, (a, b) in enumerate(shapes)]
    layout = PackedLayout.from_adapter(LitheAdapter(modules=mods, indices={}, rank=16, alpha=16.0))
    assert layout.num_indices == 3_589_632
    assert layout.num_words * 8 == 4_102_440


def test_decode_packed_reference_matches_quant():
    ad = random_adapter(SMALL, rank=16)
    words, layout = pack_adapter(ad)
    pa, pb = cb.twin_pools()
    out = decode_packed_reference(words, layout, pa, pb)
    got = split_decoded(out, layout)
    for name, (a, b) in ref_factors(ad).items():
        assert torch.equal(got[name][0], a.to(torch.bfloat16))
        assert torch.equal(got[name][1], b.to(torch.bfloat16))


def test_decode_packed_falls_back_on_cpu():
    ad = random_adapter(SMALL, rank=8, seed=3)
    words, layout = pack_adapter(ad)
    pa, pb = cb.twin_pools()
    assert torch.equal(decode_packed(words, layout, pa, pb), decode_packed_reference(words, layout, pa, pb))


def kohya_like_adapter(seed=0, d=64, mlp=128, rank=4):
    mods = [("lora_unet_double_blocks_0_img_attn_qkv", d, 3 * d),
            ("lora_unet_double_blocks_0_txt_mlp_0", d, mlp),
            ("lora_unet_double_blocks_0_img_mod_lin", d, 6 * d),
            ("lora_unet_single_blocks_0_linear1", d, 3 * d + mlp),
            ("lora_unet_single_blocks_0_linear2", d + mlp, d)]
    return random_adapter(mods, rank=rank, seed=seed, key_format="kohya")


def test_kohya_split_paths():
    ad = kohya_like_adapter()
    paths = kohya.diffusers_paths(ad)
    assert paths == ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k",
                     "transformer_blocks.0.attn.to_v", "transformer_blocks.0.ff_context.net.0.proj",
                     "transformer_blocks.0.norm1.linear", "single_transformer_blocks.0.attn.to_q",
                     "single_transformer_blocks.0.attn.to_k", "single_transformer_blocks.0.attn.to_v",
                     "single_transformer_blocks.0.proj_mlp", "single_transformer_blocks.0.proj_out"]
    with pytest.raises(KeyError):
        kohya.split_spec("lora_unet_final_layer_linear", [4, 64], [64, 4])


def test_kohya_index_split_equals_factor_split():
    ad = kohya_like_adapter(seed=5)
    by_factor = kohya.split_factors(ref_factors(ad))
    pa, pb = cb.twin_pools()
    for path, (ia, ib, sa, sb) in kohya.split_indices(ad).items():
        a = quant.decode(torch.from_numpy(ia.astype(np.int64)), sa, pa)
        b = quant.decode(torch.from_numpy(ib.astype(np.int64)), sb, pb)
        assert torch.equal(a, by_factor[path][0]) and torch.equal(b, by_factor[path][1]), path


def test_sgmv_reference_math():
    torch.manual_seed(0)
    x, a, b = torch.randn(3, 5, 32), torch.randn(3, 4, 32), torch.randn(3, 24, 4)
    s = torch.tensor([0.5, 1.0, 2.0])
    want = torch.einsum("bsd,brd,bor->bso", x, a, b) * s.view(3, 1, 1)
    torch.testing.assert_close(per_sample_lora_delta_reference(x, a, b, s), want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(per_sample_lora_delta(x, a, b, s), want, rtol=1e-5, atol=1e-5)


def test_shared_reference_matches_decode_then_matmul():
    ad = random_adapter([("l", 128, 64)], rank=16, seed=2)
    ad2 = random_adapter([("l", 128, 64)], rank=16, seed=9)
    pa, pb = cb.twin_pools()
    ia = torch.stack([torch.from_numpy(a.indices["l"]["A"].astype(np.int64)) for a in (ad, ad2)])
    ib = torch.stack([torch.from_numpy(a.indices["l"]["B"].astype(np.int64)) for a in (ad, ad2)])
    x = torch.randn(3, 7, 128)
    assign = torch.tensor([1, 0, 1])
    got = shared_codebook_lora_delta(x, ia, pa, ib, pb, assign, 16, 64, 0.5)
    facs = [ref_factors(ad)["l"], ref_factors(ad2)["l"]]
    want = torch.stack([x[i] @ facs[j][0].T @ facs[j][1].T for i, j in enumerate(assign.tolist())]) * 0.5
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)


def test_per_sample_layer_loop_equals_sgmv_route_cpu():
    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24)
    facs = [(torch.randn(4, 32), torch.randn(24, 4), 0.5), (torch.randn(4, 32), torch.randn(24, 4), 2.0)]
    loop, sgmv = PerSampleLoRALinear(base, "loop"), PerSampleLoRALinear(base, "sgmv")
    for m in (loop, sgmv):
        m.set_factors(facs)
    for x in (torch.randn(3, 6, 32), torch.randn(3, 32)):
        want = base(x) + torch.stack([
            torch.nn.functional.linear(torch.nn.functional.linear(x[i], facs[i % 2][0]), facs[i % 2][1]) * facs[i % 2][2]
            for i in range(3)])
        torch.testing.assert_close(loop(x), want, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sgmv(x), want, rtol=1e-5, atol=1e-5)


def test_lattice_search_reference_and_fast_rvq_cpu():
    blocks = torch.randn(256, 64) * 0.05
    pools = cb.build_pools(cb.SIGMA_A)
    proj = cb.projection_matrix()
    idx, sel = lattice_search_reference(blocks, proj, pools[0])
    assert torch.equal(idx, quant.nearest_index(blocks, proj))
    i_fast, r_fast = rvq_encode_blocks_fast(blocks, pools, proj)
    i_ref, r_ref = quant.rvq_encode_blocks(blocks, pools, proj)
    assert torch.equal(i_fast, i_ref) and torch.equal(r_fast, r_ref)


# ----------------------------------------------------------------------------- GPU
@gpu
def test_gpu_decode_packed_bit_exact():
    ad = random_adapter(SMALL + [("wide", 3072, 9216)], rank=16, seed=4)
    words, layout = pack_adapter(ad)
    pa, pb = (p.cuda() for p in cb.twin_pools())
    w = words.cuda()
    out = decode_packed(w, layout, pa, pb)
    assert torch.equal(out, decode_packed_reference(w, layout, pa, pb))
    got = split_decoded(out, layout)
    for name, (a, b) in ref_factors(ad, "cuda").items():
        assert torch.equal(got[name][0], a.to(torch.bfloat16))
        assert torch.equal(got[name][1], b.to(torch.bfloat16))
    idx16 = torch.from_numpy(flat_indices(ad, layout).astype(np.int16)).view(-1, layout.num_levels).cuda()
    assert torch.equal(decode_int16(idx16, layout, pa, pb), out)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@gpu
@pytest.mark.parametrize("seq,d_in,d_out,r", [(256, 3072, 9216, 16), (77, 512, 640, 16), (10, 256, 128, 4)])
def test_gpu_sgmv_matches_reference(seq, d_in, d_out, r):
    torch.manual_seed(0)
    bsz = 3
    x = (torch.randn(bsz, seq, d_in, device="cuda") * 0.5).bfloat16()
    a = (torch.randn(bsz, r, d_in, device="cuda") * 0.02).bfloat16()
    b = (torch.randn(bsz, d_out, r, device="cuda") * 0.02).bfloat16()
    s = torch.tensor([0.5, 1.0, 2.0], device="cuda").bfloat16()
    got = per_sample_lora_delta(x, a, b, s)
    ref = per_sample_lora_delta_reference(x, a, b, s)
    truth = torch.einsum("bsd,brd,bor->bso", x.float(), a.float(), b.float()) * s.float().view(-1, 1, 1)
    assert got.dtype == torch.bfloat16 and got.shape == (bsz, seq, d_out)
    assert _rel(got, truth) < 1e-2 and _rel(got, ref) < 1e-2


@gpu
@pytest.mark.parametrize("seq,d_in,d_out,r", [(256, 3072, 3072, 16), (50, 128, 192, 16), (1, 256, 384, 4)])
def test_gpu_shared_matches_reference(seq, d_in, d_out, r):
    ads = [random_adapter([("l", d_in, d_out)], rank=r, seed=s) for s in range(3)]
    pa, pb = (p.cuda() for p in cb.twin_pools())
    ia = torch.stack([torch.from_numpy(a.indices["l"]["A"].astype(np.int32)) for a in ads]).cuda()
    ib = torch.stack([torch.from_numpy(a.indices["l"]["B"].astype(np.int32)) for a in ads]).cuda()
    x = (torch.randn(4, seq, d_in, device="cuda") * 0.5).bfloat16()
    assign = torch.tensor([2, 0, 1, 2], device="cuda", dtype=torch.int32)
    got = shared_codebook_lora_delta(x, ia, pa, ib, pb, assign, r, d_out, 1.0)
    ref = shared_codebook_lora_delta_reference(x, ia, pa, ib, pb, assign, r, d_out, 1.0)
    facs = [ref_factors(a, "cuda")["l"] for a in ads]
    truth = torch.stack([x[i].float() @ facs[j][0].T @ facs[j][1].T for i, j in enumerate(assign.tolist())])
    assert got.dtype == torch.bfloat16
    assert _rel(got, ref) < 1e-2 and _rel(got, truth) < 2e-2


@gpu
def test_gpu_index_layer_matches_decoded_layer():
    ads = [random_adapter([("l", 256, 384)], rank=16, seed=s) for s in range(2)]
    base = torch.nn.Linear(256, 384).cuda().bfloat16()
    pa, pb = (p.cuda() for p in cb.twin_pools())
    ia = torch.stack([torch.from_numpy(a.indices["l"]["A"].astype(np.int32)) for a in ads]).cuda()
    ib = torch.stack([torch.from_numpy(a.indices["l"]["B"].astype(np.int32)) for a in ads]).cuda()
    idx_layer = IndexLoRALinear(base, ia, ib, pa.bfloat16(), pb.bfloat16(), 16, 1.0)
    dec_layer = PerSampleLoRALinear(base, "sgmv")
    dec_layer.set_factors([(f[0].bfloat16(), f[1].bfloat16(), 1.0)
                           for f in (ref_factors(a, "cuda")["l"] for a in ads)])
    assign = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    idx_layer.assign, dec_layer.assign = assign, assign
    facs = [ref_factors(a, "cuda")["l"] for a in ads]
    for x in (torch.randn(3, 64, 256, device="cuda").bfloat16(), torch.randn(3, 256, device="cuda").bfloat16()):
        # fp32 truth; the layer outputs are rounded to bf16 after adding the base output
        truth = torch.stack([torch.nn.functional.linear(x[i].float(), base.weight.float(), base.bias.float())
                             + x[i].float() @ facs[j][0].T @ facs[j][1].T for i, j in enumerate(assign.tolist())])
        assert _rel(idx_layer(x), truth) < 1e-2 and _rel(dec_layer(x), truth) < 1e-2


@gpu
def test_gpu_lattice_search_matches_closed_form():
    torch.manual_seed(0)
    blocks = torch.randn(8192, 64, device="cuda") * 0.02
    proj = cb.projection_matrix().cuda()
    pool = cb.build_pools(cb.SIGMA_A)[0].cuda()
    idx, rec = lattice_search(blocks, proj, pool)
    ridx, rrec = lattice_search_reference(blocks, proj, pool)
    agree = idx == ridx
    assert agree.float().mean().item() > 0.999
    assert torch.equal(rec[agree], rrec[agree])
