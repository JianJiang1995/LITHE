# Vendored vLLM SGMV kernels

`sgmv_shrink.py`, `sgmv_expand.py` and `libentry.py` are copied from the vLLM
v0.6.3 release (`vllm/lora/ops/`), the productionized Punica / S-LoRA SGMV
kernels (Chen et al., "Punica: Multi-Tenant LoRA Serving", 2023). `libentry.py`
is itself copied by vLLM from FlagGems. vLLM is licensed under Apache-2.0, the
same license as this repository; the copyright of these files stays with the
vLLM authors.

Local changes, limited to the import block of `sgmv_shrink.py` and
`sgmv_expand.py`:

* the `libentry` import points at the copy in this folder (package-relative);
* setting the environment variable `LITHE_VLLM_NO_LIBENTRY=1` replaces the
  `libentry` launch cache by an identity decorator. vLLM notes that libentry
  can be dropped from Triton 3.0 on; it only caches kernel launches and does
  not change the numerics. Use it if `libentry` fails on a newer Triton.

They are used only by `serving/bench_sgmv_kernel.py` for the head-to-head
comparison of the paper (App. "Modern paging comparison", Tab. "Batched SGMV
kernel"). The paper measurement used these kernels with torch 2.6 and Triton 3.2
(with libentry).
