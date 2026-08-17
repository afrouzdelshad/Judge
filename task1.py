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

from LLM_factory import chat_with_batch, code_augmented_chat_with
from prompt_tasks import TASK1_CODE_PROMPT, TASK1_PLAIN_PROMPT

DATA_FILE = "task1_data.csv"
KEY_FILE = "task1_key.csv"
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


def format_table(times, columns):
    """Wide, quote-free text table of all labeled series for the 'plain' prompt."""
    labels = sorted(columns)

    header = "t," + ",".join(f"{lab}_x,{lab}_y" for lab in labels)
    lines = [header]
    for i in range(len(times)):
        row = [f"{times[i]:.4g}"]
        for lab in labels:
            x, y = columns[lab][i]
            row += [f"{x:.4g}", f"{y:.4g}"]
        lines.append(",".join(row))
    return "\n".join(lines)


def load_key(path):
    """Return {output_label: original_N} from a task1_key.csv, or None if absent.

    File is a "Label,N" header followed by one row per label.
    """
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["Label"]: int(row["N"]) for row in reader if row.get("Label")}


def kendall_tau(order, true_n):
    """Kendall's tau, rescaled to [0, 1], between a predicted order and the true N values (see README)."""
    ranks = [true_n[lab] for lab in order]
    n = len(ranks)
    concordant = sum(ranks[i] < ranks[j] for i in range(n) for j in range(i + 1, n))
    return concordant / (n * (n - 1) // 2)


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


def run_plain_batch(model, labels, times, columns, n, use_batch=True):
    """Same 'plain' prompt run n times via chat_with_batch (cache [+ batch] discount)."""
    system_prompt = TASK1_PLAIN_PROMPT.format(n=len(labels))
    user_prompt = format_table(times, columns)
    print(f"[plain] querying {model} x{n} ({'batched' if use_batch else 'realtime'}) ...")
    replies = chat_with_batch(model, system_prompt, user_prompt, n=n, use_batch=use_batch)
    orders = [extract_order(r) if r else None for r in replies]
    return orders, replies


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
    p.add_argument("--key-file", default=KEY_FILE, dest="key_file")
    p.add_argument("--max-iters", type=int, default=3, dest="max_iters")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--mode", choices=["plain", "code", "both"], default="both")
    p.add_argument("--outdir", default=OUTDIR)
    p.add_argument("--scale", type=int, default=1, help="repeat the run this many times")
    p.add_argument("--no-batch", action="store_true", dest="no_batch",
                    help="skip the Anthropic Message Batches API for plain mode; fire concurrent realtime calls instead (loses the batch discount, but no queueing delay)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    times, columns = load_dataset(args.data_file)
    labels = sorted(columns)
    print(f"Loaded {len(labels)} labeled trajectories from {args.data_file}: {labels}\n")

    true_n = load_key(args.key_file)

    base_outdir = Path(args.outdir)
    scale = args.scale

    # Plain mode: one prompt, all `scale` repeats fired via chat_with_batch.
    plain_orders, plain_transcripts = [None] * scale, [None] * scale
    if args.mode in ("plain", "both"):
        plain_orders, plain_transcripts = run_plain_batch(
            args.model, labels, times, columns, scale, use_batch=not args.no_batch
        )

    trials = []
    for i in range(scale):
        if scale > 1:
            print(f"=== trial {i + 1}/{scale} ===")
        outdir = base_outdir / f"trial_{i + 1}" if scale > 1 else base_outdir
        outdir.mkdir(parents=True, exist_ok=True)
        results = {"model": args.model}

        if args.mode in ("plain", "both"):
            order, transcript = plain_orders[i], plain_transcripts[i]
            (outdir / "transcript_plain.txt").write_text(transcript or "", encoding="utf-8")
            results["plain"] = order
            if order and true_n:
                results["plain_tau"] = kendall_tau(order, true_n)
            print(f"[plain] order: {order}  tau={results.get('plain_tau')}\n")

        if args.mode in ("code", "both"):
            print(f"[code] querying {args.model} ...")
            order, transcript = run_code(args.model, labels, args.data_file, args.max_iters)
            (outdir / "transcript_code.txt").write_text(transcript, encoding="utf-8")
            results["code"] = order
            if order and true_n:
                results["code_tau"] = kendall_tau(order, true_n)
            print(f"[code] order: {order}  tau={results.get('code_tau')}\n")

        results_path = outdir / f"results_{args.model}.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to {results_path}")
        trials.append(results)

    if scale > 1:
        summary_path = base_outdir / f"summary_{args.model}.json"
        summary_path.write_text(
            json.dumps({"model": args.model, "scale": scale, "trials": trials}, indent=2), encoding="utf-8"
        )
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
