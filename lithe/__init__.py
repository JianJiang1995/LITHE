"""LITHE: Lightweight Indexed Twin Hadamard Encoding for diffusion personalisation."""
from .codebook import (BLOCK_SIZE, K, N_BITS, NUM_LEVELS, SIGMA_A, SIGMA_B, build_pools,
                       codebook_sha256, codewords, twin_pools)
from .convert import decode_factors, from_dense_lora, from_training_checkpoint, to_lora_state_dict
from .fileio import LitheAdapter, adapter_from_layers, load_lithe, read_header, save_lithe
from .module import FLUX_TARGET_MODULES, LitheLinear, TwinCodebook, inject_lithe, lithe_layers

__version__ = "1.0.0"

__all__ = [
    "BLOCK_SIZE", "K", "N_BITS", "NUM_LEVELS", "SIGMA_A", "SIGMA_B",
    "build_pools", "codebook_sha256", "codewords", "twin_pools",
    "decode_factors", "from_dense_lora", "from_training_checkpoint", "to_lora_state_dict",
    "LitheAdapter", "adapter_from_layers", "load_lithe", "read_header", "save_lithe",
    "FLUX_TARGET_MODULES", "LitheLinear", "TwinCodebook", "inject_lithe", "lithe_layers",
]
