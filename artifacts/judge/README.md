# LLM-judge prompts

* `identity_prompt.txt`: rubric for DreamBooth-30, CC101-20 and SCIdentity (SI/PF/VQ/SBD/CN). Python
  `str.format` template with `{prompt}` and `{class_name}`; doubled braces are literal.
* `sc40_prompt.txt`: rubric for SC-40 (TF/PF/VQ), with `{group}`, `{concept}`, `{group_hint}` and
  `{prompt}`.
* `sc40_group_hints.json`: the `{group_hint}` string for each SC-40 axis.

These are the exact strings used for all three judges (GPT-5.1, Claude Sonnet 4.6, Gemini 2.5 Flash).
The message is the filled template followed by the images, each preceded by a short text label:
`REFERENCE:` / `GENERATED:` for the identity rubric; `REFERENCE 1 (target concept):`,
`REFERENCE 2 (same target concept):`, `GENERATED (model output to evaluate):` for SC-40. The identity
rubric receives the subject's first training image as the reference; `{concept}` is the SC-40 concept
name with underscores (for example `candy_glass_cyber_cel`; the DoRA, VeRA, AdaLoRA, QLoRA-NF4,
LoRA-XS and T-LoRA runs passed it prefixed with the axis, `anime_style__candy_glass_cyber_cel`).

`judge.py` sends one such request to an OpenAI-compatible chat-completions endpoint, configured only
through `OPENAI_API_KEY` and `OPENAI_BASE_URL` (default `https://api.openai.com/v1`):

```bash
export OPENAI_API_KEY=...
python judge.py --rubric identity --image gen.jpg --ref train_00.jpg \
    --prompt "a photo of a <v_sub_001> cat" --class-name cat
python judge.py --rubric sc40 --image gen.jpg --ref ref1.png --ref ref2.png --group anime_style \
    --concept candy_glass_cyber_cel \
    --prompt "an anime illustration of a cat, in <v_anime_001> candy glass cyber cel style"
```

Defaults match the GPT-5.1 runs (model `gpt-5.1`, temperature 0, JSON output mode, image detail
`high`). For endpoints that reject `response_format`, pass `--no-json-mode` (the Claude and Gemini runs
used no JSON mode and no detail hint: `--no-json-mode --detail ''`). The judges are proprietary models,
so rescoring today may not reproduce the released scores exactly.
