import lzma

import numpy as np
import pytest

import lithe
from lithe import fileio


def _random_adapter(seed=0, n_modules=4):
    rng = np.random.default_rng(seed)
    modules, indices = [], {}
    for i in range(n_modules):
        name = f"transformer_blocks.{i}.attn.to_q"
        shape_a, shape_b = [16, 192], [96 + 32 * i, 16]
        modules.append({"name": name, "shape_a": shape_a, "shape_b": shape_b})
        indices[name] = {
            "A": rng.integers(0, 512, (3, 16 * 192 // 64)).astype(np.uint16),
            "B": rng.integers(0, 512, (3, -(-shape_b[0] * 16 // 64))).astype(np.uint16),
        }
    return lithe.LitheAdapter(modules=modules, indices=indices, rank=16, alpha=16.0,
                              metadata={"instance_prompt": "a photo of a <v_sub_001> cat"})


def test_lzma_stream_identity():
    raw = np.random.default_rng(1).integers(0, 512, 100_000).astype("<u2").tobytes()
    assert fileio.unpack_stream(fileio.pack_stream(raw)) == raw
    assert fileio.LZMA_PRESET == 9 | lzma.PRESET_EXTREME


def test_save_load_identity(tmp_path):
    ad = _random_adapter()
    path = tmp_path / "a.lithe"
    size = lithe.save_lithe(path, ad)
    assert size == path.stat().st_size
    ad2 = lithe.load_lithe(path)
    assert ad2.modules == ad.modules
    assert ad2.rank == 16 and ad2.alpha == 16.0 and ad2.metadata == ad.metadata
    for name in ad.indices:
        for side in ("A", "B"):
            assert ad2.indices[name][side].dtype == np.uint16
            assert np.array_equal(ad2.indices[name][side], ad.indices[name][side])
    assert ad2.raw_stream() == ad.raw_stream()
    header = lithe.read_header(path)
    assert header["codebook"]["sha256"] == lithe.codebook_sha256()
    assert header["num_indices"] == ad.num_indices
    # the index payload is the plain LZMA-9 stream of the raw uint16 indices
    blob = path.read_bytes()
    n = int.from_bytes(blob[8:12], "little")
    assert blob[12 + n:] == fileio.pack_stream(ad.raw_stream())


def test_stream_layout_all_a_then_all_b(tmp_path):
    ad = _random_adapter(n_modules=2)
    raw = np.frombuffer(ad.raw_stream(), dtype="<u2")
    names = [m["name"] for m in ad.modules]
    expect = np.concatenate([ad.indices[n]["A"].ravel() for n in names]
                            + [ad.indices[n]["B"].ravel() for n in names])
    assert np.array_equal(raw, expect)


def test_corruption_is_detected(tmp_path):
    ad = _random_adapter()
    path = tmp_path / "a.lithe"
    lithe.save_lithe(path, ad)
    blob = bytearray(path.read_bytes())
    header_end = 12 + int.from_bytes(blob[8:12], "little")
    ad.indices[ad.modules[0]["name"]]["A"][0, 0] ^= 1
    bad = tmp_path / "bad.lithe"
    bad.write_bytes(bytes(blob[:header_end]) + fileio.pack_stream(ad.raw_stream()))
    with pytest.raises(ValueError, match="checksum"):
        lithe.load_lithe(bad)
