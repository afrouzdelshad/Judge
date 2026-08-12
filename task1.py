#!/usr/bin/env python3
"""
Task 1: order labeled trajectories by number of lobes N
=========================================================

Reads task1data_shuffled.csv (a "time" column, then one column per labeled
trajectory holding "(x, y)" pairs) and asks an LLM to recover the ascending-N
order of the labels -- either from a pasted table ("plain") or by writing
Python against the CSV ("code").

Usage
-----
    python task1.py --mode both --model claude-opus-4-8

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
from prompt_tasks import TASK1_CODE_PROMPT, TASK1_PLAIN_PROMPT

DATA_FILE = "task1data_shuffled.csv"
OUTDIR = "task1_runs"


def load_dataset(path):
    """Read the CSV into (times, {label: [(x, y), ...]})."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    labels = rows[0][1:]
    times = [float(row[0]) for row in rows[1:]]
    columns = {label: [] for label in labels}
    for row in rows[1:]:
        for label, cell in zip(labels, row[1:]):
            x, y = cell.strip("()").split(",")
            columns[label].append((float(x), float(y)))
    return times, columns


def format_table(times, columns, n_points):
    """Wide, quote-free text table of all labeled series for the 'plain' prompt."""
    labels = sorted(columns)
    step = max(1, len(times) // n_points)
    idx = range(0, len(times), step)

    header = "t," + ",".join(f"{lab}_x,{lab}_y" for lab in labels)
    lines = [header]
    for i in idx:
        row = [f"{times[i]:.4g}"]
        for lab in labels:
            x, y = columns[lab][i]
            row += [f"{x:.4g}", f"{y:.4g}"]
        lines.append(",".join(row))
    return "\n".join(lines)


def extract_order(text):
    """Return the "order" list from the last ```json block that has one."""
    for block in reversed(re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "order" in data:
            return data["order"]
    return None


def run_plain(model, labels, times, columns, points):
    system_prompt = TASK1_PLAIN_PROMPT.format(n=len(labels))
    user_prompt = format_table(times, columns, points)
    reply = chat_with(model, system_prompt, user_prompt)
    return extract_order(reply), reply


def run_code(model, labels, data_file, max_iters):
    system_prompt = TASK1_CODE_PROMPT.format(
        n=len(labels), data_file=str(Path(data_file).resolve()), max_iters=max_iters
    )
    user_prompt = "Begin your analysis. Remember the protocol above."
    reply, transcript = code_augmented_chat_with(model, system_prompt, user_prompt, max_iters)
    return extract_order(reply), transcript


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", default=DATA_FILE, dest="data_file")
    p.add_argument("--points", type=int, default=120, help="rows shown per series in the 'plain' prompt")
    p.add_argument("--max-iters", type=int, default=6, dest="max_iters")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--mode", choices=["plain", "code", "both"], default="both")
    p.add_argument("--outdir", default=OUTDIR)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    times, columns = load_dataset(args.data_file)
    labels = sorted(columns)
    print(f"Loaded {len(labels)} labeled trajectories from {args.data_file}: {labels}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {"model": args.model}

    if args.mode in ("plain", "both"):
        print(f"[plain] querying {args.model} ...")
        order, transcript = run_plain(args.model, labels, times, columns, args.points)
        (outdir / "transcript_plain.txt").write_text(transcript, encoding="utf-8")
        print(f"[plain] order: {order}\n")
        results["plain"] = order

    if args.mode in ("code", "both"):
        print(f"[code] querying {args.model} ...")
        order, transcript = run_code(args.model, labels, args.data_file, args.max_iters)
        (outdir / "transcript_code.txt").write_text(transcript, encoding="utf-8")
        print(f"[code] order: {order}\n")
        results["code"] = order

    results_path = outdir / f"results_{args.model}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
