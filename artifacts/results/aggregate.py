#!/usr/bin/env python
"""Recompute the Table 1 / multi-seed appendix aggregates from the per-prompt CSVs.

For every (benchmark, method, seed) run the per-prompt rows are averaged (each benchmark has the
same number of prompts per concept, so this equals the mean over concepts of the per-concept
means). A judge average of one image is the mean of its rubric dimensions (SI/PF/VQ/SBD/CN on
the identity benchmarks, TF/PF/VQ on SC-40); judge means are computed exactly from the integer
scores. The LLM-avg of a run is the unweighted mean of its GPT-5.1, Claude Sonnet 4.6 and
Gemini 2.5 Flash judge averages. Reported values are mean and sample standard deviation over
the three runs of a method.

Default output: full-precision statistics (multi-seed appendix tables use these).
--table1: Table 1 cell convention, i.e. LLM-avg = mean of the three per-judge means and
          +- = mean of the three per-judge standard deviations, each first rounded to 2 decimals.

Usage: python aggregate.py [--dir DIR] [--table1] [--csv OUT.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

BENCHMARKS = ["dreambooth30", "cc101_20", "scidentity", "sc40"]
METHODS = ["lora", "lithe", "dora", "vera", "adalora", "qlora_nf4", "dylora", "lora_fa", "loraxs", "tlora"]
JUDGES = ["gpt51", "claude", "gemini"]
EMBEDDINGS = ["dreamsim", "clip_i", "clip_t"]
METRICS = ["llm_avg"] + [f"{j}_avg" for j in JUDGES] + EMBEDDINGS


def dims_for(bench: str) -> list[str]:
    return ["TF", "PF", "VQ"] if bench == "sc40" else ["SI", "PF", "VQ", "SBD", "CN"]


def mean(xs):
    return sum(xs) / len(xs)


def sample_std(xs) -> float:
    xs = [float(x) for x in xs]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def r2(x) -> float:
    # round-half-even on the exact value (Fraction) or the float
    return float(round(Fraction(x) if not isinstance(x, Fraction) else x, 2))


def run_means(path: Path, bench: str) -> dict:
    dims = dims_for(bench)
    acc = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], int(row["seed"]))
            for j in JUDGES:
                acc[key][f"{j}_avg"].append(Fraction(sum(int(row[f"{j}_{d}"]) for d in dims), len(dims)))
            for e in EMBEDDINGS:
                acc[key][e].append(float(row[e]))
    out = {}
    for key, cols in acc.items():
        m = {metric: mean(vals) for metric, vals in cols.items()}
        m["llm_avg"] = mean([m[f"{j}_avg"] for j in JUDGES])
        m["n"] = len(cols["gpt51_avg"])
        out[key] = m
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--table1", action="store_true", help="print Table 1 cells (LLM-avg, DreamSim)")
    parser.add_argument("--csv", type=Path, default=None, help="write the full-precision aggregate table here")
    args = parser.parse_args()

    table = []
    for bench in BENCHMARKS:
        per_run = run_means(args.dir / f"{bench}_per_prompt.csv", bench)
        print(f"\n== {bench}")
        if args.table1:
            print(f"{'method':<10} {'LLM-avg':>11} {'DreamSim':>11}")
        else:
            print(f"{'method':<10} {'seeds':<10} {'n/run':>5}  " + "  ".join(f"{m:>15}" for m in METRICS))
        for method in METHODS:
            runs = [v for (meth, _), v in sorted(per_run.items()) if meth == method]
            seeds = ",".join(str(s) for (meth, s) in sorted(per_run) if meth == method)
            if not runs:
                continue
            stats = {metric: (mean([r[metric] for r in runs]), sample_std([r[metric] for r in runs])) for metric in METRICS}
            table.append({"benchmark": bench, "method": method, "seeds": seeds, "n_per_run": runs[0]["n"],
                          **{f"{k}_{s}": round(float(v[i]), 6) for k, v in stats.items() for i, s in enumerate(["mean", "std"])}})
            if args.table1:
                llm = mean([r2(stats[f"{j}_avg"][0]) for j in JUDGES])
                llm_sd = mean([r2(stats[f"{j}_avg"][1]) for j in JUDGES])
                ds, ds_sd = stats["dreamsim"]
                print(f"{method:<10} {r2(llm):>6.2f}+-{r2(llm_sd):.2f} {ds:>6.2f}+-{ds_sd:.2f}")
            else:
                cells = []
                for metric in METRICS:
                    mu, sd = stats[metric]
                    digits = 4 if metric in EMBEDDINGS else 2
                    cells.append(f"{r2(mu) if digits == 2 else float(mu):.{digits}f}+-{sd:.{digits}f}".rjust(15))
                print(f"{method:<10} {seeds:<10} {runs[0]['n']:>5}  " + "  ".join(cells))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
