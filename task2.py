#!/usr/bin/env python3
"""
Task 2: spot the one trajectory with a different lobe count
=============================================================

Reads task2data_shuffled.csv (a "time" column, then one column per labeled
trajectory holding "(x, y)" pairs). Of the labeled trajectories, all but one
share the same number of lobes N (each from a different initial condition);
exactly one has a different N. Asks an LLM to find that outlier label --
either from a pasted table ("plain") or by writing Python against the CSV
("code").

Usage
-----
    python task2.py --mode both --model claude-opus-4-8

If task2_key.csv is present (columns: output_label,is_outlier[,original_N]),
results are scored against it; otherwise the outlier is just recorded.

Note: in "code" mode, model-written Python is exec'd locally with no
sandboxing. Only use this with trusted API providers on a machine you
control.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from LLM_factory import chat_with, code_augmented_chat_with
from prompt_tasks import TASK2_CODE_PROMPT, TASK2_PLAIN_PROMPT
from task1 import format_table, load_dataset

DATA_FILE = "task2data_shuffled.csv"
KEY_FILE = "task2_key.csv"
OUTDIR = "task2_runs"


def load_key(path):
    """Return the true outlier label from a task2_key.csv, or None if absent."""
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_outlier"].strip().lower() in ("1", "true", "yes"):
                return row["output_label"]
    return None


def extract_outlier(text):
    """Return the "outlier" label from the last ```json block that has one."""
    for block in reversed(re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "outlier" in data:
            return data["outlier"]
    return None


def run_plain(model, labels, times, columns):
    system_prompt = TASK2_PLAIN_PROMPT.format(n=len(labels), n_minus_1=len(labels) - 1)
    user_prompt = format_table(times, columns)
    reply = chat_with(model, system_prompt, user_prompt)
    return extract_outlier(reply), reply


def run_code(model, labels, data_file, max_iters):
    system_prompt = TASK2_CODE_PROMPT.format(
        n=len(labels), n_minus_1=len(labels) - 1,
        data_file=str(Path(data_file).resolve()), max_iters=max_iters,
    )
    user_prompt = "Begin your analysis. Remember the protocol above."
    reply, transcript = code_augmented_chat_with(model, system_prompt, user_prompt, max_iters)
    return extract_outlier(reply), transcript


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", default=DATA_FILE, dest="data_file")
    p.add_argument("--key-file", default=KEY_FILE, dest="key_file")
    p.add_argument("--max-iters", type=int, default=3, dest="max_iters")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--mode", choices=["plain", "code", "both"], default="both")
    p.add_argument("--outdir", default=OUTDIR)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    times, columns = load_dataset(args.data_file)
    labels = sorted(columns)
    print(f"Loaded {len(labels)} labeled trajectories from {args.data_file}: {labels}\n")

    true_outlier = load_key(args.key_file)
    if true_outlier:
        print(f"True outlier (from {args.key_file}): {true_outlier}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {"model": args.model, "true_outlier": true_outlier}

    if args.mode in ("plain", "both"):
        print(f"[plain] querying {args.model} ...")
        outlier, transcript = run_plain(args.model, labels, times, columns)
        (outdir / "transcript_plain.txt").write_text(transcript, encoding="utf-8")
        print(f"[plain] outlier: {outlier}\n")
        results["plain"] = outlier
        if true_outlier:
            results["plain_correct"] = outlier == true_outlier

    if args.mode in ("code", "both"):
        print(f"[code] querying {args.model} ...")
        outlier, transcript = run_code(args.model, labels, args.data_file, args.max_iters)
        (outdir / "transcript_code.txt").write_text(transcript, encoding="utf-8")
        print(f"[code] outlier: {outlier}\n")
        results["code"] = outlier
        if true_outlier:
            results["code_correct"] = outlier == true_outlier

    results_path = outdir / f"results_{args.model}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
