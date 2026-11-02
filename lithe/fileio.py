"""The ``.lithe`` index file: a JSON header plus one LZMA-compressed uint16 stream.

Layout::

    b"LITHE001" | uint32 LE n | xz(UTF-8 JSON header), n bytes | xz(index stream)

Both parts use ``lzma.compress`` with preset 9 | PRESET_EXTREME. The index
stream is compressed on its own, so its size is exactly the LZMA-9 payload
reported in the paper; the compressed header (module names and shapes) adds a
few KB. The index stream is the raw
little-endian uint16 index stream: the A-side indices of every module in
header order, followed by the B-side indices of every module. Each module
contributes its (L, n_blocks) index matrix in row-major order.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import codebook as cb
from . import quant

MAGIC = b"LITHE001"
LZMA_PRESET = 9 | lzma.PRESET_EXTREME


@dataclass
class LitheAdapter:
    """Index streams of one adapter plus everything needed to decode them."""

    modules: List[Dict[str, Any]]                 # [{"name", "shape_a", "shape_b"}] in stream order
    indices: Dict[str, Dict[str, np.ndarray]]      # name -> {"A": (L, nA) uint16, "B": (L, nB) uint16}
    rank: int
    alpha: float
    key_format: str = "diffusers"                  # "diffusers" or "kohya" module naming
    base_model: str = "black-forest-labs/FLUX.1-dev"
    block_size: int = cb.BLOCK_SIZE
    n_bits: int = cb.N_BITS
    num_levels: int = cb.NUM_LEVELS
    sigma_a: List[float] = field(default_factory=lambda: list(cb.SIGMA_A))
    sigma_b: List[float] = field(default_factory=lambda: list(cb.SIGMA_B))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_indices(self) -> int:
        return sum(v["A"].size + v["B"].size for v in self.indices.values())

    def raw_stream(self) -> bytes:
        names = [m["name"] for m in self.modules]
        parts = [self.indices[n]["A"] for n in names] + [self.indices[n]["B"] for n in names]
        return b"".join(np.ascontiguousarray(p, dtype="<u2").tobytes() for p in parts)


def pack_stream(raw: bytes) -> bytes:
    return lzma.compress(raw, preset=LZMA_PRESET)


def unpack_stream(blob: bytes) -> bytes:
    return lzma.decompress(blob)


def _header(ad: LitheAdapter, raw: bytes) -> Dict[str, Any]:
    return {
        "format": "lithe",
        "version": 1,
        "codebook": {
            "name": "hadamard9-twin",
            "block_size": ad.block_size,
            "n_bits": ad.n_bits,
            "num_levels": ad.num_levels,
            "sigma_a": list(ad.sigma_a),
            "sigma_b": list(ad.sigma_b),
            "sha256": cb.codebook_sha256(ad.block_size, ad.n_bits),
        },
        "rank": ad.rank,
        "alpha": ad.alpha,
        "key_format": ad.key_format,
        "base_model": ad.base_model,
        "index_dtype": "uint16-le",
        "num_indices": ad.num_indices,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "modules": ad.modules,
        "metadata": ad.metadata,
    }


def save_lithe(path, ad: LitheAdapter) -> int:
    """Write ``ad`` to ``path``; returns the file size in bytes."""
    if ad.n_bits > 16:
        raise ValueError("uint16 streams need n_bits <= 16")
    raw = ad.raw_stream()
    header = pack_stream(json.dumps(_header(ad, raw), separators=(",", ":")).encode("utf-8"))
    blob = pack_stream(raw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        f.write(blob)
    return path.stat().st_size


def read_header(path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not a .lithe file")
        (n,) = struct.unpack("<I", f.read(4))
        return json.loads(unpack_stream(f.read(n)).decode("utf-8"))


def load_lithe(path, verify: bool = True) -> LitheAdapter:
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not a .lithe file")
        (n,) = struct.unpack("<I", f.read(4))
        header = json.loads(unpack_stream(f.read(n)).decode("utf-8"))
        raw = unpack_stream(f.read())
    cbh = header["codebook"]
    if verify:
        if hashlib.sha256(raw).hexdigest() != header["payload_sha256"]:
            raise ValueError(f"{path}: payload checksum mismatch")
        if cbh["sha256"] != cb.codebook_sha256(cbh["block_size"], cbh["n_bits"]):
            raise ValueError(f"{path}: codebook hash mismatch")
    stream = np.frombuffer(raw, dtype="<u2")
    L, bs = cbh["num_levels"], cbh["block_size"]
    modules = header["modules"]
    counts_a = [L * quant.num_blocks(m["shape_a"], bs) for m in modules]
    counts_b = [L * quant.num_blocks(m["shape_b"], bs) for m in modules]
    if sum(counts_a) + sum(counts_b) != stream.size:
        raise ValueError(f"{path}: stream length {stream.size} does not match header")
    indices: Dict[str, Dict[str, np.ndarray]] = {}
    off = 0
    for m, c in zip(modules, counts_a):
        indices[m["name"]] = {"A": stream[off:off + c].reshape(L, -1).copy()}
        off += c
    for m, c in zip(modules, counts_b):
        indices[m["name"]]["B"] = stream[off:off + c].reshape(L, -1).copy()
        off += c
    return LitheAdapter(
        modules=modules, indices=indices, rank=header["rank"], alpha=header["alpha"],
        key_format=header["key_format"], base_model=header.get("base_model", ""),
        block_size=bs, n_bits=cbh["n_bits"], num_levels=L,
        sigma_a=cbh["sigma_a"], sigma_b=cbh["sigma_b"], metadata=header.get("metadata", {}),
    )


def adapter_from_layers(layers, rank: int, alpha: float, key_format: str = "diffusers",
                        base_model: str = "black-forest-labs/FLUX.1-dev",
                        metadata: Optional[Dict[str, Any]] = None) -> LitheAdapter:
    """Run a final search on every ``LitheLinear`` and collect the index streams."""
    modules, indices = [], {}
    for name, layer in layers.items():
        idx = layer.indices()
        modules.append({"name": name, "shape_a": list(layer.lora_A.shape),
                        "shape_b": list(layer.lora_B.shape)})
        indices[name] = {k: v.cpu().numpy().astype(np.uint16) for k, v in idx.items()}
    book = next(iter(layers.values())).codebook
    metadata = {"module_set": f"{key_format}-{len(modules)}", **(metadata or {})}
    return LitheAdapter(modules=modules, indices=indices, rank=rank, alpha=float(alpha),
                        key_format=key_format, base_model=base_model,
                        block_size=book.block_size, n_bits=book.n_bits,
                        num_levels=len(book.sigma_a), sigma_a=list(book.sigma_a),
                        sigma_b=list(book.sigma_b), metadata=metadata or {})
