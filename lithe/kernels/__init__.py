"""Triton kernels behind the paper's serving and search numbers, with PyTorch references.

Every kernel has a pure-PyTorch reference with the same math that runs on any
device, and each public entry point falls back to that reference when Triton
is not installed or the input is not on a CUDA device.

* ``packed``  9-bit index packing (7 indices per uint64 word) and the fused
              whole-adapter decode from the packed GPU-resident form.
* ``sgmv``    two-stage per-sample LoRA delta (shrink, expand) over decoded factors.
* ``shared``  index-resident per-sample LoRA delta: the decode is fused into the
              shrink / expand tiles, reading one codebook pool shared by all samples.
* ``search``  closed-form nearest-codeword search (one Hadamard sign per bit).
* ``layers``  ``nn.Module`` wrappers that route each sample of a batch to its own
              adapter with the kernels above.
"""
from ._triton import HAS_TRITON
from .layers import IndexLoRALinear, PerSampleLoRALinear, wrap_linears
from .packed import (PackedLayout, decode_packed, decode_packed_reference, pack_adapter,
                     pack_indices, split_decoded, unpack_indices)
from .search import lattice_search, lattice_search_reference, rvq_encode_blocks_fast
from .sgmv import per_sample_lora_delta, per_sample_lora_delta_reference
from .shared import shared_codebook_lora_delta, shared_codebook_lora_delta_reference

__all__ = [
    "HAS_TRITON",
    "IndexLoRALinear", "PerSampleLoRALinear", "wrap_linears",
    "PackedLayout", "decode_packed", "decode_packed_reference", "pack_adapter",
    "pack_indices", "split_decoded", "unpack_indices",
    "lattice_search", "lattice_search_reference", "rvq_encode_blocks_fast",
    "per_sample_lora_delta", "per_sample_lora_delta_reference",
    "shared_codebook_lora_delta", "shared_codebook_lora_delta_reference",
]
