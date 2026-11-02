# Per-prompt evaluation results

Prompt-level judge scores and embedding metrics behind Table 1 (`tab:db30-pareto`) and the
multi-seed appendix tables (`tab:multiseed-db30`, `-cc101`, `-scidentity`, `-sc40`). One row is one
generated image (one method, one training run, one concept, one prompt). No images are included.

| file | rows | content |
|---|---:|---|
| `dreambooth30_per_prompt.csv` | 4,500 | 30 subjects x 5 prompts x 10 methods x 3 runs |
| `cc101_20_per_prompt.csv` | 3,000 | 20 subjects x 5 prompts x 10 methods x 3 runs |
| `scidentity_per_prompt.csv` | 3,000 | 20 subjects x 5 prompts x 10 methods x 3 runs |
| `sc40_per_prompt.csv` | 5,520 | 40 concepts x 5 prompts x 8 methods x 3 runs, plus 40 x 3 prompts x 2 methods x 3 runs (DyLoRA, LoRA-FA) |
| `vanilla_flux_per_prompt.csv` | 550 | FLUX.1-dev without adapter, one run, GPT-5.1 judge only |
| `aggregate.py` | | recomputes the Table 1 and appendix aggregates from the CSVs |

## Columns

| column | meaning |
|---|---|
| `benchmark` | `dreambooth30`, `cc101_20`, `scidentity` or `sc40` |
| `method` | `lora`, `lithe`, `dora`, `vera`, `adalora`, `qlora_nf4`, `dylora`, `lora_fa`, `loraxs`, `tlora` (Table 1 rows LoRA, LITHE, DoRA, VeRA, AdaLoRA, QLoRA-NF4, DyLoRA, LoRA-FA, LoRA-XS, T-LoRA); `vanilla_flux` in the vanilla file |
| `seed` | training seed of the run. LoRA and LITHE: 42, 1, 7. The other adapters: 42, 43, 44. |
| `prompt_set` | which prompt list the image was generated from; join with `../prompts/<benchmark>.csv` on (`prompt_set`, `concept_id`, `prompt_idx`) |
| `group` | SC-40 only: axis (`outfit`, `lighting`, `pose`, `painter_style`, `anime_style`) |
| `concept_id` | subject or concept name, same as in `../prompts/` |
| `prompt_idx` | 0-based index of the prompt in its list |
| `gpt51_*`, `claude_*`, `gemini_*` | integer 1-5 rubric scores of GPT-5.1, Claude Sonnet 4.6 and Gemini 2.5 Flash. Identity benchmarks: `SI` subject identity, `PF` prompt following, `VQ` visual quality, `SBD` subject-background disentanglement, `CN` class naturalness. SC-40: `TF` target fidelity, `PF`, `VQ` |
| `<judge>_avg` | mean of that judge's rubric dimensions for the image (5 dims on identity, 3 on SC-40) |
| `llm_avg` | mean of `gpt51_avg`, `claude_avg`, `gemini_avg` for the image |
| `dreamsim` | DreamSim similarity to the reference images: max over references of the cosine similarity of DreamSim embeddings (higher is closer) |
| `clip_i` | CLIP ViT-L/14 image-image cosine similarity, mean over the reference images |
| `clip_t` | CLIP ViT-L/14 cosine similarity between the image and its generation prompt (the prompt text includes the trigger token) |

References: on the identity benchmarks the embedding metrics use all training images of the subject
and the judges see the first training image; on SC-40 the judges and the embedding metrics use the
same two reference images of the concept. Floating-point columns are rounded to 6 decimals.

The vanilla file has `rubric` instead of the three-judge block. `rubric = identity` / `sc40` rows are
scored with the rubrics of `../judge/`. The DreamBooth-30 and CC101-20 vanilla rows (`rubric =
legacy_si_pf_vq`) were scored with an earlier GPT-5.1 rubric that returned only SI, PF and VQ, on the
`vanilla_sks` prompts, and their `gpt51_avg` is the mean of those three scores.

## Prompt sets

| `prompt_set` | used by |
|---|---|
| `trigger` | identity benchmarks: LoRA, LITHE, DyLoRA, LoRA-FA; vanilla FLUX on SCIdentity |
| `legacy_sks` | identity benchmarks: DoRA, VeRA, AdaLoRA, QLoRA-NF4, LoRA-XS, T-LoRA (trained and prompted with the DreamBooth `sks` token) |
| `vanilla_sks` | vanilla FLUX on DreamBooth-30 and CC101-20 |
| `main` | SC-40: LoRA, LITHE, DoRA, VeRA, AdaLoRA, QLoRA-NF4, LoRA-XS, T-LoRA, vanilla FLUX |
| `short3` | SC-40: DyLoRA, LoRA-FA (three prompts per concept) |

## Aggregation

For each (benchmark, method, seed) run, average a metric over all rows of the run (every concept has
the same number of prompts, so this equals the mean over concepts of the per-concept means). For a
run, LLM-avg = (GPT-avg + Claude-avg + Gemini-avg) / 3. Then over the three runs of a method:

* Multi-seed appendix tables: mean and sample standard deviation (n - 1) of GPT-avg, DreamSim, CLIP-I,
  CLIP-T, Gemini-avg and Claude-avg over the three runs.
* Table 1, LLM-avg: the mean of the three per-judge means and, after the +-, the mean of the three
  per-judge standard deviations, each per-judge value first rounded to two decimals as printed in the
  appendix tables. The +- is therefore the average judge-level seed spread, not the standard deviation
  of the per-run LLM-avg (the latter is `llm_avg` in the default output of `aggregate.py`).
* Table 1, DreamSim: mean +- sample standard deviation over the three runs, two decimals.
* Vanilla FLUX (Table 1 first row): a single run scored by GPT-5.1 only; its LLM-avg column is the
  GPT-5.1 average, and DreamSim is the mean over the images.

```bash
python aggregate.py            # full-precision multi-seed statistics per benchmark and method
python aggregate.py --table1   # Table 1 LLM-avg and DreamSim cells
python aggregate.py --csv agg.csv
```

## Check against the paper

Recomputed from these CSVs with `aggregate.py`:

* Multi-seed appendix tables: 240 / 240 printed cells (10 methods x 4 benchmarks x 6 metrics) are
  reproduced exactly.
* Table 1 LLM-avg: 40 / 40 cells (mean and +-) reproduced with the convention above. With full
  precision instead, the means agree for 36 / 40 cells and differ by 0.01 for the other four
  (LITHE DreamBooth-30 4.4150 -> 4.41 vs 4.42 printed; DoRA DreamBooth-30, DoRA CC101-20,
  QLoRA-NF4 SCIdentity); the standard deviation of the per-run LLM-avg agrees with the printed +- in
  25 / 40 cells.
* Table 1 DreamSim: 35 / 40 cells reproduced. The other five differ by 0.01 because the table rounds
  the four-decimal appendix value a second time (DoRA DreamBooth-30 mean 0.4949 printed .50; VeRA
  DreamBooth-30 std 0.0247 printed .03; VeRA SCIdentity mean 0.4846 printed .49; T-LoRA SCIdentity
  mean 0.5348 printed .54; LITHE SCIdentity std 0.0146 printed .02).
* Vanilla FLUX: GPT-5.1 average 3.33 / 3.37 / 3.98 / 3.53 and DreamSim 0.41 / 0.43 / 0.43 / 0.42 on
  DreamBooth-30 / CC101-20 / SCIdentity / SC-40, as printed.

LoRA and LITHE (Table 1, three runs each):

| benchmark | LoRA LLM-avg | LITHE LLM-avg | LoRA DreamSim | LITHE DreamSim |
|---|---|---|---|---|
| DreamBooth-30 | 4.56 +- .02 | 4.42 +- .02 | .67 +- .01 | .60 +- .01 |
| CC101-20 | 4.19 +- .04 | 4.14 +- .02 | .62 +- .00 | .56 +- .00 |
| SCIdentity | 4.48 +- .02 | 4.30 +- .04 | .69 +- .01 | .65 +- .01 (printed .02) |
| SC-40 | 3.94 +- .03 | 3.77 +- .02 | .55 +- .00 | .53 +- .00 |

Full-precision LLM-avg (mean +- sample std of the per-run LLM-avg): LoRA 4.5637 +- 0.0168,
4.1938 +- 0.0259, 4.4833 +- 0.0179, 3.9363 +- 0.0098; LITHE 4.4150 +- 0.0163, 4.1364 +- 0.0229,
4.3031 +- 0.0437, 3.7720 +- 0.0208 (DreamBooth-30, CC101-20, SCIdentity, SC-40).

## Judging setup

GPT-5.1 was called through the OpenAI chat-completions API at temperature 0 with JSON output mode,
image detail `high`, and `max_completion_tokens` 300 (identity) / 400 (SC-40). Claude Sonnet 4.6 and
Gemini 2.5 Flash were called through an OpenAI-compatible endpoint at temperature 0 with the same
prompt text and image order, without JSON mode; when a reply omitted a score field, the missing
fields were requested once more with a short score-only follow-up on the same images. Each image was
scored once per judge. Prompts and a minimal re-implementation are in `../judge/`.
