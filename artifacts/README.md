# Paper artifacts

Evaluation artifacts for the FLUX.1-dev benchmarks of the paper (Table 1 and the multi-seed appendix
tables). Text and CSV only; no images, checkpoints or API credentials.

```
artifacts/
  prompts/   per-concept evaluation prompt lists (DreamBooth-30, CC101-20, SCIdentity, SC-40)
  results/   per-prompt judge scores and embedding metrics, plus aggregate.py (see results/README.md)
  judge/     verbatim LLM-judge prompt templates and a minimal judge script (see judge/README.md)
```

## prompts/

One CSV per benchmark, one row per (prompt set, concept, prompt index).

* `dreambooth30.csv`, `cc101_20.csv`, `scidentity.csv`: `benchmark, prompt_set, concept_id, trigger,
  class_name, prompt_idx, prompt`. `concept_id` is the subject folder name of the public DreamBooth /
  CustomConcept101 release (SCIdentity: the synthetic subject name). `trigger` is the per-concept token
  (`<v_sub_NNN>`, `<v_sub_CNNN>`, `<v_id_NNN>`), or `sks` for the legacy lists.
* `sc40.csv`: `benchmark, prompt_set, concept_num, group, concept_id, concept_label, trigger,
  prompt_idx, prompt`, with `group` the SC-40 axis and `trigger` the per-concept token
  (`<v_outfit_NNN>`, `<v_light_NNN>`, `<v_pose_NNN>`, `<v_pstyle_NNN>`, `<v_anime_NNN>`).

Each concept has five evaluation prompts (`trigger`, `legacy_sks`, `vanilla_sks`, SC-40 `main`) or
three (SC-40 `short3`). `results/README.md` lists which methods use which prompt set; LoRA and
LITHE use `trigger` on the identity benchmarks and `main` on SC-40.
