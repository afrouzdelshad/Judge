#!/usr/bin/env python3
"""
Task 2b: spot every trajectory with a different lobe count
=============================================================

Reads task2b_data.csv (a "time" column, then one column per labeled
trajectory holding "(x, y)" pairs). Most of the labeled trajectories share
the same majority lobe count N (each from a different initial condition);
the rest have an N that differs from that majority. Asks an LLM to say how
many outliers there are and identify each one from a pasted table.

Usage
-----
    python task2b.py --model claude-opus-4-8

If task2b_key.csv is present ("Label,N" header, one row per true outlier
label), results are scored against it; otherwise the outliers are just
recorded.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from LLM_factory import chat_with_batch
from prompt_tasks import TASK2B_PLAIN_PROMPT
from task1 import format_table, load_dataset

DATA_FILE = "task2b_data.csv"
KEY_FILE = "task2b_key.csv"
OUTDIR = "task2b_runs"


def load_key(path):
    """Return the set of true outlier labels from a task2b_key.csv, or None.

    File is a "Label,N" header followed by one row per true outlier label
    (unlike task2a_key.csv, majority labels are simply omitted).
    """
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["Label"] for row in reader if row.get("Label")}


def extract_outliers(text):
    """Return the "outliers" list from the last ```json block that has one."""
    for block in reversed(re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "outliers" in data and isinstance(data["outliers"], list):
            return data["outliers"]
    return None


def f1_score(predicted, true_outliers):
    """F1 between a predicted and true outlier-label set, in [0, 1] (1 = exact match)."""
    pred_set = set(predicted)
    tp = len(pred_set & true_outliers)
    precision = tp / len(pred_set) if pred_set else (1.0 if not true_outliers else 0.0)
    recall = tp / len(true_outliers) if true_outliers else (1.0 if not pred_set else 0.0)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def run_plain_batch(model, labels, times, columns, n, use_batch=False):
    """Same 'plain' prompt run n times via chat_with_batch (cache [+ batch] discount)."""
    system_prompt = TASK2B_PLAIN_PROMPT.format(n=len(labels))
    user_prompt = format_table(times, columns)
    print(f"[plain] querying {model} x{n} ({'batched' if use_batch else 'realtime'}) ...")
    replies = chat_with_batch(model, system_prompt, user_prompt, n=n, use_batch=use_batch)
    outliers = [extract_outliers(r) if r else None for r in replies]
    return outliers, replies


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", default=DATA_FILE, dest="data_file")
    p.add_argument("--key-file", default=KEY_FILE, dest="key_file")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--outdir", default=OUTDIR)
    p.add_argument("--scale", type=int, default=1, help="repeat the run this many times")
    p.add_argument("--batch", action="store_true",
                    help="use the Anthropic Message Batches API for plain mode instead of concurrent realtime calls (50%% cheaper, but queues with no latency SLA)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    times, columns = load_dataset(args.data_file)
    labels = sorted(columns)
    print(f"Loaded {len(labels)} labeled trajectories from {args.data_file}: {labels}\n")

    true_outliers = load_key(args.key_file)
    if true_outliers is not None:
        print(f"True outliers (from {args.key_file}): {sorted(true_outliers)}\n")

    base_outdir = Path(args.outdir)
    scale = args.scale

    plain_outliers, plain_transcripts = run_plain_batch(
        args.model, labels, times, columns, scale, use_batch=args.batch
    )

    trials = []
    for i in range(scale):
        if scale > 1:
            print(f"=== trial {i + 1}/{scale} ===")
        outdir = base_outdir / f"trial_{i + 1}" if scale > 1 else base_outdir
        outdir.mkdir(parents=True, exist_ok=True)
        results = {"model": args.model, "true_outliers": sorted(true_outliers) if true_outliers is not None else None}

        outliers, transcript = plain_outliers[i], plain_transcripts[i]
        (outdir / "transcript_plain.txt").write_text(transcript or "", encoding="utf-8")
        print(f"[plain] outliers: {outliers}\n")
        results["plain"] = outliers
        if outliers is not None and true_outliers is not None:
            results["plain_f1"] = f1_score(outliers, true_outliers)

        results_path = outdir / f"results_{args.model}.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to {results_path}")
        trials.append(results)

    if scale > 1:
        summary = {"model": args.model, "scale": scale, "trials": trials}
        for key in ("plain_f1",):
            scored = [t[key] for t in trials if key in t]
            if scored:
                summary[f"{key}_mean"] = sum(scored) / len(scored)

        summary_path = base_outdir / f"summary_{args.model}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
