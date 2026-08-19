#!/usr/bin/env python3
"""
Task 4: estimate the lobe count N of every trajectory
=========================================================

Reads task4_data.csv (a "time" column, then one column per labeled
trajectory holding "(x, y)" pairs). Each trajectory has an independently,
randomly chosen lobe count N. Asks an LLM to estimate N for every
trajectory from a pasted table.

Usage
-----
    python task4.py --model claude-opus-4-8

If task4_key.csv is present ("Label,N" header, one row per label), results
are scored against it; otherwise the estimates are just recorded.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from LLM_factory import chat_with_batch
from prompt_tasks import TASK4_PLAIN_PROMPT
from task1 import format_table, load_dataset

DATA_FILE = "task4_data.csv"
KEY_FILE = "task4_key.csv"
OUTDIR = "task4_runs"


def load_key(path):
    """Return {label: N} from a task4_key.csv ("Label,N" header), or None."""
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["Label"]: int(row["N"]) for row in reader if row.get("Label")}


def extract_estimates(text):
    """Return the "estimates" dict from the last ```json block that has one."""
    for block in reversed(re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("estimates"), dict):
            return data["estimates"]
    return None


def score_estimates(estimates, true_n):
    """Mean per-label relative accuracy, in [0, 1] (1 = every N guessed exactly).

    Each label's score is max(0, 1 - |pred - true| / true), so accuracy is
    scaled by how many lobes there actually are -- an error of 2 on a true
    N of 2 is much worse than the same error on a true N of 20. Labels
    missing from `estimates` (or unparseable) score 0, so omissions are
    penalized rather than silently ignored.

    Also returns the raw mean absolute error over the labels that were
    actually predicted (or None if none were).
    """
    per_label, abs_errors = {}, []
    for label, true in true_n.items():
        pred = (estimates or {}).get(label)
        try:
            pred = int(pred)
        except (TypeError, ValueError):
            pred = None
        if pred is None:
            per_label[label] = 0.0
            continue
        err = abs(pred - true)
        abs_errors.append(err)
        per_label[label] = max(0.0, 1.0 - err / true)

    accuracy = sum(per_label.values()) / len(true_n) if true_n else 1.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else None
    return accuracy, mae, per_label


def run_plain_batch(model, labels, times, columns, n, use_batch=False):
    """Same 'plain' prompt run n times via chat_with_batch (cache [+ batch] discount)."""
    system_prompt = TASK4_PLAIN_PROMPT.format(n=len(labels))
    user_prompt = format_table(times, columns)
    print(f"[plain] querying {model} x{n} ({'batched' if use_batch else 'realtime'}) ...")
    replies = chat_with_batch(model, system_prompt, user_prompt, n=n, use_batch=use_batch)
    estimates = [extract_estimates(r) if r else None for r in replies]
    return estimates, replies


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

    true_n = load_key(args.key_file)

    base_outdir = Path(args.outdir)
    scale = args.scale

    plain_estimates, plain_transcripts = run_plain_batch(
        args.model, labels, times, columns, scale, use_batch=args.batch
    )

    trials = []
    for i in range(scale):
        if scale > 1:
            print(f"=== trial {i + 1}/{scale} ===")
        outdir = base_outdir / f"trial_{i + 1}" if scale > 1 else base_outdir
        outdir.mkdir(parents=True, exist_ok=True)
        results = {"model": args.model}

        estimates, transcript = plain_estimates[i], plain_transcripts[i]
        (outdir / "transcript_plain.txt").write_text(transcript or "", encoding="utf-8")
        results["plain"] = estimates
        if estimates is not None and true_n is not None:
            accuracy, mae, per_label = score_estimates(estimates, true_n)
            results["plain_accuracy"], results["plain_mae"], results["plain_per_label"] = accuracy, mae, per_label
        print(f"[plain] estimates: {estimates}  accuracy={results.get('plain_accuracy')}  mae={results.get('plain_mae')}\n")

        results_path = outdir / f"results_{args.model}.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to {results_path}")
        trials.append(results)

    if scale > 1:
        summary = {"model": args.model, "scale": scale, "trials": trials}
        for key in ("plain_accuracy", "plain_mae"):
            scored = [t[key] for t in trials if t.get(key) is not None]
            if scored:
                summary[f"{key}_mean"] = sum(scored) / len(scored)

        summary_path = base_outdir / f"summary_{args.model}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
