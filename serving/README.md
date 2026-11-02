# Serving kernels and benchmarks

This folder holds the benchmark harness behind the serving results of the
paper (Sec. 4.3 "Production-serving wall-time", Sec. 5.2 "Same-batch
heterogeneous routing", and the appendix sections on index residency, the
hybrid sweep, the serving-kernel ablation, the serial-fuse re-measurement and
the modern paging comparison). The Triton kernels themselves live in
[`lithe/kernels/`](../lithe/kernels/) and are importable on their own:

| Module | Kernel | Paper |
|---|---|---|
| `lithe.kernels.packed` | 9-bit index packing (7 indices per uint64) and fused whole-adapter decode from the packed GPU-resident form | 4.56 ms decode of all 304 Kohya linears; 4.10 MB logical / 4.19 MB measured per resident slot |
| `lithe.kernels.sgmv` | two-stage per-sample LoRA delta (shrink, expand) over decoded factors | same-batch routing over decoded factors; head-to-head with the vLLM SGMV kernel |
| `lithe.kernels.shared` | index-resident per-sample LoRA delta, decode fused into the shrink / expand tiles with one codebook pool shared by all samples | LITHE column of the wall-time table (with `torch.compile`) and the serving-kernel ablation |
| `lithe.kernels.search` | closed-form nearest-codeword search (nine Hadamard signs, one launch) | per-block search microbenchmark (52 us vs. 135 us for `cdist`) |
| `lithe.kernels.layers` | `PerSampleLoRALinear` (decoded factors, `loop` or `sgmv`) and `IndexLoRALinear` (index slots) | wrappers used by every benchmark here |

Every kernel has a PyTorch reference in the same module and falls back to it
when Triton is missing or the tensors are on the CPU. `tests/test_kernels.py`
checks the kernels against the references on CUDA (skipped without a GPU) and
the references against `lithe.quant` on the CPU.

## Setup

* One NVIDIA GPU with 80 GB (the paper used A100-SXM4-80GB). The decode, SGMV
  and lattice microbenchmarks need a few GB; everything that loads FLUX.1-dev
  needs about 34 GB for the bf16 pipeline plus the serving working set.
* Python environment of the release (`pip install -e ".[flux,test]"`) plus
  `triton` (it ships with the CUDA builds of PyTorch on Linux).
* FLUX.1-dev in diffusers format (`--pretrained`, or `$FLUX_MODEL_PATH`).
* Adapters: the released `.lithe` files from
  [Fanjue/LITHE](https://huggingface.co/Fanjue/LITHE) (110 rank-16 adapters,
  Kohya-304 layout). Pass a folder with `--adapter-dir`; it is searched
  recursively for `*.lithe`, and the scripts take the first N in sorted order.

```bash
huggingface-cli download Fanjue/LITHE --local-dir ./adapters
export FLUX_MODEL_PATH=./checkpoints/FLUX.1-dev
```

The Kohya layout fuses QKV and contains the modulation linears. On the
diffusers `FluxTransformer2DModel` one adapter therefore covers 494 linears;
`lithe/kohya.py` maps the 304 Kohya modules onto them, splitting B (and the
index streams, which are block aligned at every split point) by rows.

All scripts print a log and, with `--out file.json`, write every measurement
with the software versions. Times are CUDA-synchronised wall-clock; memory is
PyTorch `memory_allocated` / `memory_reserved` (not NVML).

## Benchmarks and the paper numbers they reproduce

### Fused decode and cold swap (`bench_decode.py`, no FLUX weights)

```bash
python serving/bench_decode.py --adapter-dir ./adapters/indices/dreambooth30 --trials 200
```

| Quantity | Paper (A100 80 GB) | Where |
|---|---|---|
| fused decode of one packed adapter (304 Kohya linears), warm | 4.56 ms mean, p99 4.59 ms, sigma 0.012 ms | App. "N_max co-resident inventory stress test and decode microbench" |
| packed vs. int16 decode | bit-exact (max abs diff 0) | same |
| raw int16 blob from disk to decoded factors | about 10 ms | App. "Engineered cold-swap path" |

The script also times the per-module PyTorch decode loop and the full
`.lithe` (LZMA) cold path for reference, and checks the fused output against
`lithe.decode_factors` cast to bf16 (expected max abs diff 0).

### Bit-packed residency and end-to-end hot-swap (`bench_residency.py`)

```bash
# Tab. "Allocator accounting ..." and Tab. "End-to-end serving from a 10,000-adapter resident pool"
python serving/bench_residency.py --adapter-dir ./adapters/indices/dreambooth30 \
    --n-slots 10000 --renders 3 --size 1024 --steps 28 --out residency.json
# OOM boundary (App. "N_max ..."): grows the bank in tiers, then bisects
python serving/bench_residency.py --adapter-dir ./adapters/indices/dreambooth30 --probe-oom
```

| Quantity | Paper |
|---|---|
| FLUX.1-dev bf16 pipeline, allocated | 33.76 GB |
| after 10,000 packed slots, buffers and wrappers | 75.70 GB (increment 41.94 GB, 4.19 MB per slot) |
| int16 attach at 11,000 slots / 12,000 slots | fits under 80 GB / OOM at 84.5 GB |
| allocator-sensitive bisection estimate | 11,764 slots |
| hot-swap (fused decode + Kohya->diffusers split + install), first / warm | 450.18 ms (Triton JIT) / 9.55 ms, 9.36 ms |
| 1024^2 / 28-step render after the swap | 14.87 s / 14.06 s / 14.06 s |

### Hybrid N_res x B_serv sweep (`bench_hybrid.py`)

```bash
python serving/bench_hybrid.py --adapter-dir ./adapters/indices/dreambooth30 \
    --points 11000x1 8000x4 4000x8 1000x14 4x14 --size 512 --steps 16 --out hybrid.json
# P6, the 1024^2 / 28-step anchor of the main-text table
python serving/bench_hybrid.py --adapter-dir ./adapters/indices/dreambooth30 \
    --points 10000x14 --size 1024 --steps 28 --out hybrid_p6.json
```

Paper (Tab. "Hybrid N_res x B_serv operating points" and its appendix version;
512^2 / 16 steps except P6; peak = PyTorch allocated/reserved):

| Point | N_res | B_serv | VRAM peak | decode + install | forward | ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 11,000 | 1 | 80.87 GB | 10.5 ms | 2.17 s | 2,169 |
| P2 | 8,000 | 4 | 70.26 GB | 30.4 ms | 8.11 s | 2,029 |
| P3 | 4,000 | 8 | 56.24 GB | 58.0 ms | 15.88 s | 1,985 |
| P4 | 1,000 | 14 | 47.81 GB | 100.6 ms | 27.49 s | 1,963 |
| P5 | 4 | 14 | 43.65 GB | 111.7 ms | 27.47 s | 1,962 |
| P6 (1024^2 / 28) | 10,000 | 14 | 75.70 GB | | | 13,040 |

The paper used 8 distinct DreamBooth adapters (cycled when B > 8); `--n-real`
sets this.

### Same-batch heterogeneous routing (`bench_same_batch.py`)

```bash
# LITHE: index-resident shared-codebook kernel + torch.compile (paper default)
python serving/bench_same_batch.py --adapter-dir ./adapters --mode lithe-index --compile default \
    --module-set attn-ff --batch 14 --size 1024 --steps 28 --trials 3
# the same without compile (serving-kernel ablation, "shared cache, no compile")
python serving/bench_same_batch.py --adapter-dir ./adapters --mode lithe-index --compile none ...
# decoded factors + Triton SGMV, no compile (paging-table LITHE row, B = 96)
python serving/bench_same_batch.py --adapter-dir ./adapters --mode lithe-decoded --kernel sgmv \
    --batch 96 --size 256 --steps 4
# resident dense LoRA columns: --mode dense with --lora-dir (or --dense-from-lithe), --kernel loop,
# --compile none (LoRA+SGMV column) or --compile default (LoRA+compile column)
```

Tab. "Same-batch wall-time" (3-trial mean +- std, A100 80 GB):

| Config | Serial fuse | LoRA+SGMV (resident) | LoRA+compile (resident) | LITHE |
|---|---:|---:|---:|---:|
| 256^2 / 4 steps, B = 96 | 150.0 +- 1.9 s | 18.30 s | 15.59 +- 0.13 s | 11.00 +- 0.01 s |
| 512^2 / 16 steps, B = 14 | 43.3 +- 0.3 s | 27.51 s | 19.28 +- 0.20 s | 18.08 +- 0.01 s |
| 1024^2 / 28 steps, B = 14 | 195.3 +- 1.7 s | 188.27 s | 132.4 +- 0.16 s | 133.4 +- 0.03 s |

Tab. "Component ablation of LITHE's default serving kernel" (1024^2 / 28,
B = 14): decoded SGMV without shared cache or compile 179.00 s (`--mode
lithe-decoded --kernel sgmv`); shared cache without compile 180.06 +- 0.24 s
(`--mode lithe-index`); shared cache + compile 133.4 +- 0.03 s (`--mode
lithe-index --compile default`).
App. "Modern paging comparison" (256^2 / 4, B = 96, 494 linears): LITHE
same-batch routing 16.30 +- 0.08 s and dense factors through the same SGMV
kernel 16.26 +- 0.12 s, both measured with decoded factors and the Triton SGMV
kernel without compile (`--mode lithe-decoded --kernel sgmv`, and `--mode dense
--kernel sgmv`); an earlier pass of the same configuration gave 16.43 s.

### Serial and paging baselines (`bench_paging.py`)

```bash
# "Serial fuse" column (load -> fuse -> render -> unfuse -> unload per request)
python serving/bench_paging.py --adapter-dir ./adapters --from-lithe --tier serial-disk \
    --n 96 --size 256 --steps 4 --evict-cmd 'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches'
# Tab. "Serving-path comparison", B = 96 distinct adapters, 256^2 / 4 steps
for t in serial-pinned async-prefetch hotswap hotswap-steelman; do
  python serving/bench_paging.py --adapter-dir ./adapters --from-lithe --tier $t --n 96 --out paging_$t.json
done
```

| Serving path (B = 96, 256^2 / 4) | Paper wall |
|---|---:|
| serial CPU-pinned (`serial-pinned`) | 59.16 +- 0.30 s |
| async pipelined pager, transfer hidden (`async-prefetch`) | 56.88 +- 0.35 s |
| official diffusers hotswap under torch.compile (`hotswap`) | 73.79 +- 0.58 s |
| idealised hotswap, conversion amortised (`hotswap-steelman`) | 41.57 +- 0.06 s |
| disk serial fuse, cold network file system / warm page cache / local NVMe (`serial-disk`) | 163.5 s / 172.8 +- 2.4 s / 154.7 +- 0.9 s |

### SGMV kernel head-to-head (`bench_sgmv_kernel.py`, no FLUX weights)

```bash
python serving/bench_sgmv_kernel.py --adapter-dir ./adapters/indices/dreambooth30 --batch 96 --seq 256
```

Paper (Tab. "Batched SGMV kernel"): one LoRA-delta pass over the 304 FLUX
layer shapes at B = 96, seq 256: ours 195.0 ms, vLLM release of the Punica /
S-LoRA SGMV kernel 473.9 ms; worst cosine 1.0000, max relative error 1e-4.
The vLLM kernels are vendored in `third_party/vllm_sgmv/` (see its README).

### Hot-swap latency (`bench_hotswap.py`)

```bash
python serving/bench_hotswap.py --adapter-dir ./adapters/indices/dreambooth30 --n-resident 8 24 40
```

Paper: pointer hot-swap of pre-decoded factors over all wrapped linears 1.171 ms
mean, p99 1.240 ms over 200 swaps, independent of the number of resident
adapters (App. "N_max ..."); decode-on-swap from the packed bank about 9.5 ms
warm; engineered cold swap from a raw int16 blob about 10 ms.

## Check of this harness on the released adapters

Measured with the scripts above on one A100-SXM4-80GB (torch 2.6.0+cu124,
Triton 3.2.0, diffusers 0.37.1, the software stack of the paper's wall-time
runs), on the released DreamBooth-30 / SC-40 adapters; the host was not
exclusive (other GPUs busy). GPU kernel tests: `tests/test_kernels.py`, 19
passed.

| Quantity | This harness | Paper |
|---|---:|---:|
| fused packed decode, 304 linears, warm (`bench_decode`) | 3.00 ms mean, p99 3.06 ms | 4.56 ms, p99 4.59 ms |
| packed vs. int16 vs. PyTorch decode (bf16) | max abs diff 0 / 0 | 0 |
| one SGMV pass over 304 FLUX shapes, B = 96 (`bench_sgmv_kernel`): ours / vLLM / bmm | 191.4 / 461.5 / 495.5 ms | 195.0 / 473.9 ms / - |
| ours vs. vLLM numerics | worst cosine 1.000000, rel. mean error 6e-5 | 1.0000, 1e-4 |
| FLUX.1-dev base; 10,000 packed slots (`bench_residency`) | 33.76 GB; 75.70 GB (4.195 MB/slot) | 33.76 GB; 75.70 GB (4.19 MB/slot) |
| hot-swap from the 10,000-slot pool, first / warm | 242.6 ms / 9.35 ms, 9.21 ms | 450.18 ms / 9.55 ms, 9.36 ms |
| 1024^2 / 28-step render after the swap | 15.6 s / 15.2 s / 15.2 s | 14.87 s / 14.06 s / 14.06 s |
| B = 96, 256^2 / 4: `lithe-index`, attn-ff, compile | 11.16 +- 0.07 s | 11.00 +- 0.01 s |
| B = 96, 256^2 / 4: `lithe-index`, attn-ff, no compile | 15.33 +- 0.03 s | - |
| B = 96, 256^2 / 4: `lithe-decoded --kernel sgmv`, 494 linears | 16.19 +- 0.01 s | 16.30 +- 0.08 s |
| B = 96, 256^2 / 4: `dense --kernel loop`, attn-ff | 20.80 +- 0.32 s | 18.30 s (one trial) |
| pointer hot-swap over 494 linears (`bench_hotswap`) | 2.69 ms mean | 1.171 ms |
| decode-on-swap from packed slot / cold raw int16 blob (`bench_hotswap`) | 7.86 ms / 10.29 ms | ~9.5 ms / ~10 ms |
| per request, `serial-pinned`: fuse / unfuse / H2D | 0.092 / 0.092 / 0.017 s | 0.092 / 0.092 / 0.017 s |
| per request, `async-prefetch`: exposed wait / hidden page-in | 0.000 / 0.015 s | 0.000 / 0.015 s |
| per request, `hotswap` load / `hotswap-steelman` load | 0.39-0.65 s / 0.018 s | 0.350 s / 0.015 s |

B = 96 used 70 distinct adapters (the DreamBooth-30 and SC-40 files), cycled;
the paper used 96 distinct ones. The dense row decodes the LITHE adapters into
dense factors, while the paper used separately trained LoRA files. The paging
tiers were only smoke-tested with 8 requests (no B = 96 walls). The pointer
hot-swap of this wrapper is about 2x slower than the paper's figure; we have
not profiled the difference.

## What differs from the paper setup

* **Adapters.** All benchmarks run on the released Kohya-304 `.lithe` adapters.
  The paper's residency, hybrid, decode and hot-swap runs used the same Kohya
  index streams. The wall-time table and the serving-kernel ablation used
  adapters trained with the diffusers trainer on the 342 attention / FFN
  linears; `--module-set attn-ff` restricts the Kohya adapters to those 342
  linears (the modulation linears and `proj_mlp` / `proj_out` are dropped), which
  matches the wrapped-layer count but not the exact adapters.
* **Dense LoRA baselines.** The paper used independently trained rank-16 LoRA
  files (SC-40 plain LoRA, Kohya and diffusers keys). Here `--dense-from-lithe`
  / `--from-lithe` decode the LITHE adapters into ordinary bf16 LoRA factors of
  the same shapes, so compute, transfer sizes and file sizes match, while the
  adapter values differ; pass `--lora-dir` to use your own LoRA files.
* **"LoRA+SGMV" column.** The two resident dense-LoRA columns of the wall-time
  table were measured with the per-sample loop wrapper (`--kernel loop`), not
  with the Triton SGMV kernel; the "dense factors resident, same SGMV kernel"
  row of the paging table used the Triton SGMV kernel (`--kernel sgmv`).
* **Wrapper of the residency / hybrid runs.** The paper's residency, end-to-end
  hot-swap and hybrid runs routed samples with the per-sample loop wrapper; this
  is the default (`--kernel loop`) of `bench_residency.py` and `bench_hybrid.py`.
* **Fused decode.** The kernel gathers rows of the fp32 pool tables (K = 512
  codewords x 64 per level, rebuilt from the Hadamard construction at start-up)
  and sums the L = 3 levels in fp32 registers; this is the closed-form decode
  of the paper evaluated through a 128 KB lookup table per level and side.
* **Lattice search.** The paper adapters were trained with the PyTorch search;
  the Triton search kernel was benchmarked in isolation. `rvq_encode_blocks_fast`
  is a drop-in for `lithe.quant.rvq_encode_blocks`, not the trainer default.
* **Serial-fuse storage.** The paper's serial-fuse column read adapters from a
  network file system with a system-wide page / inode cache eviction before
  every request (excluded from the timed window), on the host with the lowest
  CPU load among four. `--evict-cmd` reproduces the eviction; the storage tier
  and host load of your machine will move this number most.
* **Software.** Paper wall-time and paging numbers were measured with torch
  2.6 (CUDA 12.4), Triton 3.2 and diffusers 0.37.1; some appendix runs with
  torch 2.10 (CUDA 12.8). Triton and `torch.compile` versions change absolute
  times, and the first call of every kernel includes JIT compilation and
  autotuning (the scripts warm up before timing).
* **Not ported.** The N = 500 capacity-regime experiment of App. "Modern paging
  comparison" (LRU host pager vs. packed residency, B = 64 / 96) and the
  delta-precompute pager variant (which does not fit in 80 GB) depend on the
  original experiment harness and are not included; the routing-leakage pixel
  check of App. "Same-batch per-sample routing" can be repeated with
  `bench_same_batch.py --save-images`.
