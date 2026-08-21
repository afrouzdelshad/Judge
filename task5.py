#!/usr/bin/env python3
"""
Task 5: evaluate pairwise complexity/chaos claims about two trajectories
=============================================================================

Reads task5{a,b,c}_data.csv (same "time" + one "(x, y)" column per label
layout as task1/task2/task3/task4, but with exactly two labels, A and B) and
asks an LLM to act as a scientific evaluator over six candidate claims about
the relative complexity and chaos of the pair, selecting all that it
considers supported by the data, from a pasted table.

--variant a: A and B have similar complexity but different levels of chaos.
--variant b: A and B have similar levels of chaos but different complexity.
--variant c: A and B differ in both complexity and chaos.

Usage
-----
    python task5.py --variant a --model claude-opus-4-8
    python task5.py --variant b --model claude-opus-4-8
    python task5.py --variant c --model claude-opus-4-8

If the matching key file is present ("Answer" header, one row per correct
claim number), results are scored against it; otherwise the selected claims
are just recorded.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from LLM_factory import chat_with_batch
from prompt_tasks import TASK5_PLAIN_PROMPT
from task1 import format_table, load_dataset

DATA_FILES = {"a": "task5a_data.csv", "b": "task5b_data.csv", "c": "task5c_data.csv"}
KEY_FILES = {"a": "task5a_key.csv", "b": "task5b_key.csv", "c": "task5c_key.csv"}
OUTDIRS = {"a": "task5a_runs", "b": "task5b_runs", "c": "task5c_runs"}


def load_key(path):
    """Return the set of true claim numbers from a task5{a,b,c}_key.csv, or None.

    File is an "Answer" header followed by one row per correct claim number.
    """
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {int(row["Answer"]) for row in reader if row.get("Answer")}


def extract_claims(text):
    """Return the "claims" list from the last ```json block that has one."""
    for block in reversed(re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("claims"), list):
            try:
                return [int(c) for c in data["claims"]]
            except (TypeError, ValueError):
                continue
    return None


def f1_score(predicted, true_claims):
    """F1 between a predicted and true claim-number set, in [0, 1] (1 = exact match)."""
    pred_set = set(predicted)
    tp = len(pred_set & true_claims)
    precision = tp / len(pred_set) if pred_set else (1.0 if not true_claims else 0.0)
    recall = tp / len(true_claims) if true_claims else (1.0 if not pred_set else 0.0)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def run_plain_batch(model, labels, times, columns, n, use_batch=False):
    """Same 'plain' prompt run n times via chat_with_batch (cache [+ batch] discount)."""
    system_prompt = TASK5_PLAIN_PROMPT
    user_prompt = format_table(times, columns)
    print(f"[plain] querying {model} x{n} ({'batched' if use_batch else 'realtime'}) ...")
    replies = chat_with_batch(model, system_prompt, user_prompt, n=n, use_batch=use_batch)
    claims = [extract_claims(r) if r else None for r in replies]
    return claims, replies


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["a", "b", "c"], required=True)
    p.add_argument("--data-file", default=None, dest="data_file")
    p.add_argument("--key-file", default=None, dest="key_file")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--outdir", default=None)
    p.add_argument("--scale", type=int, default=1, help="repeat the run this many times")
    p.add_argument("--batch", action="store_true",
                    help="use the Anthropic Message Batches API for plain mode instead of concurrent realtime calls (50%% cheaper, but queues with no latency SLA)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_file = args.data_file or DATA_FILES[args.variant]
    key_file = args.key_file or KEY_FILES[args.variant]
    outdir_default = args.outdir or OUTDIRS[args.variant]

    times, columns = load_dataset(data_file)
    labels = sorted(columns)
    print(f"Loaded {len(labels)} labeled trajectories from {data_file}: {labels}\n")

    true_claims = load_key(key_file)
    if true_claims is not None:
        print(f"True claims (from {key_file}): {sorted(true_claims)}\n")

    base_outdir = Path(outdir_default)
    scale = args.scale

    plain_claims, plain_transcripts = run_plain_batch(
        args.model, labels, times, columns, scale, use_batch=args.batch
    )

    trials = []
    for i in range(scale):
        if scale > 1:
            print(f"=== trial {i + 1}/{scale} ===")
        outdir = base_outdir / f"trial_{i + 1}" if scale > 1 else base_outdir
        outdir.mkdir(parents=True, exist_ok=True)
        results = {
            "model": args.model,
            "variant": args.variant,
            "true_claims": sorted(true_claims) if true_claims is not None else None,
        }

        claims, transcript = plain_claims[i], plain_transcripts[i]
        (outdir / f"transcript_plain_{args.model}.txt").write_text(transcript or "", encoding="utf-8")
        print(f"[plain] claims: {claims}\n")
        results["plain"] = claims
        if claims is not None and true_claims is not None:
            results["plain_f1"] = f1_score(claims, true_claims)

        results_path = outdir / f"results_{args.model}.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to {results_path}")
        trials.append(results)

    if scale > 1:
        summary = {"model": args.model, "variant": args.variant, "scale": scale, "trials": trials}
        for key in ("plain_f1",):
            scored = [t[key] for t in trials if key in t]
            if scored:
                summary[f"{key}_mean"] = sum(scored) / len(scored)

        summary_path = base_outdir / f"summary_{args.model}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
