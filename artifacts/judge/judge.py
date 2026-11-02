#!/usr/bin/env python
"""Score one generated image with the LITHE paper's LLM-judge rubrics.

Calls an OpenAI-compatible chat-completions endpoint. Configuration comes only from the
environment: OPENAI_API_KEY (required) and OPENAI_BASE_URL (optional, default
https://api.openai.com/v1). Standard library only.

Identity rubric (DreamBooth-30, CC101-20, SCIdentity): one reference image, scores SI/PF/VQ/SBD/CN.
  python judge.py --rubric identity --image gen.jpg --ref ref_00.jpg \
      --prompt "a photo of a <v_sub_001> cat" --class-name cat

Concept-transfer rubric (SC-40): two reference images, scores TF/PF/VQ.
  python judge.py --rubric sc40 --image gen.jpg --ref ref_a.png --ref ref_b.png \
      --prompt "an anime illustration of a cat, in <v_anime_001> candy glass cyber cel style" \
      --group anime_style --concept candy_glass_cyber_cel

Prints one JSON object with the integer scores, their mean ("avg") and the judge's free-text fields.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIMS = {"identity": ["SI", "PF", "VQ", "SBD", "CN"], "sc40": ["TF", "PF", "VQ"]}
# settings of the GPT-5.1 runs in the paper
MAX_COMPLETION_TOKENS = {"identity": 300, "sc40": 400}


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"


def image_part(path: str, detail: str | None) -> dict:
    url = {"url": data_url(path)}
    if detail:
        url["detail"] = detail
    return {"type": "image_url", "image_url": url}


def build_content(args) -> list[dict]:
    if args.rubric == "identity":
        if len(args.ref) != 1:
            raise SystemExit("--rubric identity takes exactly one --ref")
        text = (HERE / "identity_prompt.txt").read_text(encoding="utf-8").rstrip("\n")
        text = text.format(prompt=args.prompt, class_name=args.class_name)
        return [
            {"type": "text", "text": text},
            {"type": "text", "text": "REFERENCE:"},
            image_part(args.ref[0], args.detail),
            {"type": "text", "text": "GENERATED:"},
            image_part(args.image, args.detail),
        ]
    if len(args.ref) != 2:
        raise SystemExit("--rubric sc40 takes exactly two --ref")
    hints = json.loads((HERE / "sc40_group_hints.json").read_text(encoding="utf-8"))
    hint = hints.get(args.group, "the target concept demonstrated by the two reference images")
    text = (HERE / "sc40_prompt.txt").read_text(encoding="utf-8").rstrip("\n")
    text = text.format(group=args.group, concept=args.concept, group_hint=hint, prompt=args.prompt)
    return [
        {"type": "text", "text": text},
        {"type": "text", "text": "REFERENCE 1 (target concept):"},
        image_part(args.ref[0], args.detail),
        {"type": "text", "text": "REFERENCE 2 (same target concept):"},
        image_part(args.ref[1], args.detail),
        {"type": "text", "text": "GENERATED (model output to evaluate):"},
        image_part(args.image, args.detail),
    ]


def parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])


def call(args, content: list[dict]) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("set OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = {"model": args.model, "messages": [{"role": "user", "content": content}], "temperature": 0}
    if not args.no_json_mode:
        body["response_format"] = {"type": "json_object"}
        body["max_completion_tokens"] = MAX_COMPLETION_TOKENS[args.rubric]
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last = None
    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                reply = json.loads(response.read().decode("utf-8"))
            parsed = parse_json(reply["choices"][0]["message"]["content"])
            scores = {d: int(parsed[d]) for d in DIMS[args.rubric]}
            if not all(1 <= s <= 5 for s in scores.values()):
                raise ValueError(f"score out of range: {scores}")
            out = {**scores, "avg": sum(scores.values()) / len(scores), "reason": parsed.get("reason", "")}
            if args.rubric == "sc40":
                out["description"] = parsed.get("description", "")
            out["model"] = args.model
            return out
        except Exception as exc:  # noqa: BLE001 - retry API and parse failures
            last = exc
            if attempt < args.retries - 1:
                time.sleep(2**attempt + 2)
    raise SystemExit(f"judge failed after {args.retries} attempts: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rubric", choices=sorted(DIMS), required=True)
    parser.add_argument("--image", required=True, help="generated image")
    parser.add_argument("--ref", action="append", required=True, help="reference image (identity: 1, sc40: 2)")
    parser.add_argument("--prompt", required=True, help="generation prompt")
    parser.add_argument("--class-name", default="subject", help="identity rubric: subject class word")
    parser.add_argument("--group", default="", help="sc40 rubric: outfit|lighting|pose|painter_style|anime_style")
    parser.add_argument("--concept", default="", help="sc40 rubric: concept name")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--detail", default="high", help="image detail hint; pass '' to omit")
    parser.add_argument("--no-json-mode", action="store_true",
                        help="omit response_format / max_completion_tokens (for endpoints that do not support them)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    args.detail = args.detail or None
    print(json.dumps(call(args, build_content(args)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
