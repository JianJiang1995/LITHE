# LITHE: Lightweight Indexed Twin Hadamard Encoding for Diffusion Personalization

Reference code for the NeurIPS 2026 paper (to appear) by Jian Jiang, Oya Celiktutan, Yaohui Wang and Yutong Ban.

Code: https://github.com/JianJiang1995/LITHE. Pretrained index files: https://huggingface.co/Fanjue/LITHE (coming soon).

## What LITHE is

Diffusion personalization deployments serve growing catalogues of LoRA adapters, but resident-LoRA serving is capped near the GPU memory budget (about 150 rank-16 adapters on an A100 80 GB GPU running FLUX.1-dev). LITHE is a LoRA-compatible deployment representation that stores each adapter as an integer-index stream over a deterministic Hadamard codebook. The same on-disk indices drive two serving modes: an index-resident mode that routes a heterogeneous batch of adapters in one forward, and a decoded mode in which the indices are expanded into an ordinary LoRA. On disk, the rank-16 FLUX.1-dev adapters evaluated in the paper (Kohya-style, 304 fused linears including the modulation layers) average 3.23 MB as an LZMA-9 index stream, about 47x / 190x smaller than the matched bf16 r=16 / r=64 LoRA (153.3 / 612.6 MB). The default 342-module diffusers configuration of this release (attention and FFN linears, no modulation layers) averages 1.93 MB.

Each LoRA factor A or B is cut into 64-value blocks. Every block is stored as three 9-bit codeword IDs (L = 3 residual levels, K = 512) that index a fixed codebook derived from the Sylvester-Hadamard matrix. A and B use the same codewords with different scales (the "twin" codebook): sigma_A = 0.06 on all levels and sigma_B = (0.0025, 0.001, 0.0005). Adapters are trained natively in the discrete space: fp32 proxy factors are quantised in the forward pass and updated through a straight-through estimator, with lazy re-quantisation every 10 forwards. The codebook is a procedural constant (0 KB per adapter, nothing learned), and the decode of a block is a closed-form sum of scaled Hadamard sign vectors.

## Repository layout

```
lithe/
  codebook.py   deterministic Hadamard-9bit twin codebook (byte-exact, hashable)
  quant.py      closed-form twin RVQ encode / decode, straight-through estimator
  module.py     LitheLinear wrapper and injection into FLUX.1-dev (diffusers) linears
  fileio.py     .lithe index file: uint16 indices, LZMA-9 packing, checksums
  convert.py    .lithe <-> dense LoRA state dict; import of original training checkpoints
  serve.py      index-resident multi-adapter batched forward (PyTorch reference)
scripts/
  train_lithe_flux.py   train a LITHE adapter on an image folder (FLUX.1-dev)
  encode.py             dense LoRA or training checkpoint -> .lithe
  decode_to_lora.py     .lithe -> standard LoRA safetensors (stock diffusers loader)
  infer.py              generate images with a .lithe file
  serve_demo.py         heterogeneous adapters in one batched FLUX forward
  cpu_smoke.py          end-to-end toy pipeline on CPU, no FLUX weights
configs/flux_lithe_default.yaml   paper default hyperparameters
tests/                  pytest suite, CPU only
artifacts/              evaluation prompts, per-prompt results, judge prompts
```

## Install

```bash
git clone https://github.com/JianJiang1995/LITHE.git lithe && cd lithe
pip install -e .            # core: codebook, quantiser, index files, CPU tests
pip install -e ".[flux,test]"  # + diffusers / transformers / peft for FLUX.1-dev, and pytest
# or exactly the reference environment:
pip install -r requirements.txt
```

Reference environment: Python 3.10+, torch 2.10.0 (CUDA 12.8), diffusers 0.37.1, transformers 4.57.6, peft 0.18.1, accelerate 1.12.0, safetensors 0.7.0.

### FLUX.1-dev weights

The base model is not distributed here. Accept the FLUX.1 [dev] Non-Commercial License on the Hugging Face model page [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), log in, and either pass the repo id directly or download a local copy:

```bash
huggingface-cli login
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ./checkpoints/FLUX.1-dev
export FLUX_MODEL_PATH=./checkpoints/FLUX.1-dev   # optional; scripts also take --pretrained
```

Adapters trained on FLUX.1-dev are derivatives of FLUX.1-dev and are subject to its license.

## Quickstart

### 1. CPU check (no GPU, no weights)

```bash
pytest -q
python scripts/cpu_smoke.py
```

`cpu_smoke.py` trains LITHE on a toy two-layer network for 300 STE steps, writes a `.lithe` file, reloads it, decodes it to a dense LoRA and checks that the dense LoRA reproduces the LITHE forward. Expected tail of the output:

```
"eval_mse_after": ...       (about half of eval_mse_before)
"dense_vs_lithe_max_abs_diff": 0.0
"indices_roundtrip_identical": true
CPU SMOKE PASS
```

With diffusers and peft installed, `tests/test_diffusers_integration.py` also builds a tiny random `FluxTransformer2DModel` on CPU and checks that a decoded LITHE adapter loaded through the stock `FluxLoraLoaderMixin.load_lora_weights` reproduces the LITHE forward, that the FLUX flow-matching loss runs through the wrappers, and that the batched multi-adapter forward equals per-adapter forwards.

### 2. Train an adapter

A DreamBooth-style folder of 3 to 20 images of one subject or style:

```bash
python scripts/train_lithe_flux.py \
  --pretrained ./checkpoints/FLUX.1-dev \
  --instance_data_dir ./data/my_subject \
  --instance_prompt "a photo of a <v_sub_001> cat" \
  --output_dir ./outputs/my_subject
```

Defaults come from `configs/flux_lithe_default.yaml` (rank 16, alpha 16, AdamW at 1e-4 constant, 1500 steps, batch 1 x 4 accumulation, 1024 px, logit-normal timesteps, lazy re-quantisation every 10 forwards). Any of `--max_train_steps`, `--resolution`, `--rank`, `--learning_rate`, `--gradient_accumulation_steps`, `--seed`, `--save_every`, `--gradient_checkpointing` overrides the config. The script runs on a single GPU that holds the bf16 FLUX transformer plus activations (the paper used one 80 GB A100); `--gradient_checkpointing` reduces activation memory.

Output: `outputs/my_subject/adapter.lithe`. With the default 342 target linears and r = 16 it holds 2,101,248 indices and is about 1.8 to 2.0 MB.

### 3. Decode to a standard LoRA

```bash
python scripts/decode_to_lora.py --lithe outputs/my_subject/adapter.lithe --out outputs/my_subject/lora.safetensors
```

The output (about 90 MB in bf16 at r = 16) is a plain PEFT/diffusers LoRA (`transformer.<module>.lora_A.weight` / `lora_B.weight`, alpha / r folded into B):

```python
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("outputs/my_subject/lora.safetensors")
```

### 4. Generate

```bash
python scripts/infer.py --pretrained ./checkpoints/FLUX.1-dev \
  --lithe outputs/my_subject/adapter.lithe \
  --prompt "a photo of a <v_sub_001> cat on a beach" --out outputs/samples
```

`--mode peft` (default) decodes in memory and uses `load_lora_weights`; `--mode native` wraps the linears with `LitheLinear` and decodes from the index streams. Defaults: 28 steps, guidance 3.5, 1024 x 1024, seed 42.

### 5. Several adapters in one forward

```bash
python scripts/serve_demo.py --pretrained ./checkpoints/FLUX.1-dev \
  --lithe subject.lithe --lithe style.lithe \
  --request "0|a photo of a <v_sub_001> cat" \
  --request "1|a lighthouse in <v_sty_001> style" \
  --request "0:0.7,1:0.7|a <v_sub_001> cat in <v_sty_001> style" \
  --request "none|a lighthouse"
```

All adapters stay resident as index streams; each forward decodes only the adapters used by the batch and applies the rank-stacked per-sample update of the paper (`lithe/serve.py`). This is a readable PyTorch reference of the routing; the fused Triton training-search and SGMV serving kernels used for the paper's timing numbers are not part of this release, so wall-time results are not reproduced by this code.

### 6. Encode an existing LoRA (post hoc)

```bash
python scripts/encode.py --lora some_flux_lora.safetensors --out some_flux_lora.lithe
```

This quantises a continuous LoRA after training. It is lossier than native STE training, which is the method evaluated in the paper. `--checkpoint` imports index checkpoints written by the original research trainers.

## Codebook specification

With 0-based bit positions (LSB = bit 0):

* `H_syl[t, j] = (-1)^popcount(t AND j)` for `t, j` in 0..63 (order-64 Sylvester-Hadamard matrix).
* `U = H_syl[:, 0:9]`, a 64 x 9 matrix of +-1 with `U^T U = 64 I`.
* For index `k` in 0..511: `s_k[q] = +1` if bit `q` of `k` is set, else `-1`.
* Codeword `c_k = U s_k / 3`, a 64-vector with entries in {-3, -7/3, ..., 7/3, 3} and unit per-coordinate variance.
* Pools: `pool_A[l] = 0.06 c` for l = 1, 2, 3 and `pool_B[l] = (0.0025, 0.001, 0.0005)[l] c`.

The exact integer table `3 c_k` (int8, 512 x 64, row-major) has SHA-256 `eb1f4de0d64df9980436232299238e56ff04efa9ab075ded3f72c2a821ad0301` (`lithe.codebook_sha256()`). The sign convention above is the one of the trained checkpoints; writing `s_k[q] = (-1)^(bit q of k)` instead is the same codebook with indices relabelled `k -> 511 - k`.

All codewords have norm 8, so the nearest codeword to a residual `r` has bit `q` set exactly when `<r, U[:, q]> > 0`. Encoding is greedy over the three levels; `tests/test_quant.py` checks that this closed form equals `cdist`/`argmin` search over the same pools. The search runs in fp32 (bf16 rounding can flip bits near the hyperplanes).

## Index file format (`.lithe`)

```
b"LITHE001" | uint32 LE n | xz(JSON header) (n bytes) | xz(index stream)
```

The index stream is little-endian uint16: the A-side indices of all modules in header order, then the B-side indices, each module as its (L, n_blocks) matrix in row-major order. Factors are flattened row-major and zero-padded to a multiple of 64 before blocking. Both parts use `lzma.compress(preset=9 | PRESET_EXTREME)`. The header stores the module names and factor shapes, rank, alpha, the codebook parameters and hash, and a SHA-256 of the raw stream; `lithe.load_lithe` verifies both hashes.

## Pretrained index files

Coming soon (TODO): the 110 rank-16 adapters evaluated in the paper (DreamBooth-30, CC101-20, SCIdentity, SC-40) will be published on Hugging Face at [Fanjue/LITHE](https://huggingface.co/Fanjue/LITHE) as `.lithe` files; the repository is not populated yet. They use the Kohya 304-module layout (header `module_set = kohya-304`), decode with `scripts/decode_to_lora.py` to Kohya-keyed LoRA files that `FluxPipeline.load_lora_weights` loads, and require FLUX.1-dev.

## Paper artifacts

`artifacts/` holds the evaluation data behind Table 1 and the multi-seed appendix tables (text and CSV only, no images):

* `artifacts/prompts/`: per-concept evaluation prompt lists for DreamBooth-30, CC101-20, SCIdentity and SC-40 (concept id, trigger, class, prompt index, prompt text).
* `artifacts/results/`: per-prompt scores of every Table 1 row and run (all rubric dimensions of GPT-5.1, Claude Sonnet 4.6 and Gemini 2.5 Flash, their consensus, DreamSim, CLIP-I, CLIP-T; vanilla FLUX is scored by GPT-5.1 only), with `aggregate.py`, which recomputes the table entries. Columns, aggregation and the check against the paper are in `artifacts/results/README.md`.
* `artifacts/judge/`: the verbatim judge prompt templates and `judge.py`, a minimal judge for one image against an OpenAI-compatible API (`OPENAI_API_KEY`, `OPENAI_BASE_URL`).

## Scope of this release and notes on reproduction

* Included: the codebook, native STE training on FLUX.1-dev (diffusers), index files, decoding to standard LoRA, inference, and a PyTorch reference of index-resident multi-adapter routing.
* Data: the synthetic benchmarks SCIdentity and SC-40 are not part of this release; their data-generation protocol is described in the paper appendix. DreamBooth and CustomConcept101 are available from their original authors.
* Not included: the fused Triton kernels, the serving / wall-time benchmark harness, the cross-architecture (SDXL, Qwen-Image) trainers, and evaluation scripts. The LLM-judge prompts and a minimal judge script are in `artifacts/judge/` (the judges are paid proprietary APIs, so rescoring may drift from the released scores). TODO: release DreamSim / CLIP evaluation scripts.
* Some paper tables were produced with a Kohya sd-scripts network module that applies the same codebook to the fused FLUX linears (304 modules including the modulation linears). `scripts/encode.py --checkpoint` imports such index files and `scripts/decode_to_lora.py` writes them with Kohya keys, which `load_lora_weights` converts. TODO: port of the Kohya training module.
* Training differs from a numerical replay of the paper runs in implementation details (single-process loop instead of accelerate, per-module drift trigger). The quantiser, codebook, STE and schedule follow the paper.

## Citation

```bibtex
@inproceedings{jiang2026lithe,
  title     = {{LITHE}: Lightweight Indexed Twin Hadamard Encoding for Diffusion Personalization},
  author    = {Jiang, Jian and Celiktutan, Oya and Wang, Yaohui and Ban, Yutong},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {To appear. Code: https://github.com/JianJiang1995/LITHE}
}
```

## License

Code: Apache-2.0 (see `LICENSE`). FLUX.1-dev weights and adapters derived from them are governed by the FLUX.1 [dev] Non-Commercial License.
