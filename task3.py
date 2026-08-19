#!/usr/bin/env python3
"""
Task 3: cluster labeled trajectories by inferred complexity level
=====================================================================

Reads task3a_data.csv / task3b_data.csv (same "time" + one "(x, y)" column
per label layout as task1/task2) and asks an LLM to partition the labels
into groups that share the same lobe count N, ordered from least to most
complex, from a pasted table.

--variant a (fixed group structure): the number of groups (X) and members
per group (Y) are read from task3a_key.csv and disclosed to the model in
the prompt; only the group *membership* is withheld.

--variant b (unknown group structure): neither X nor Y is disclosed; the
model must also report its own estimates of X and Y. Group sizes need not
be uniform.

Usage
-----
    python task3.py --variant a --model claude-opus-4-8
    python task3.py --variant b --model claude-opus-4-8

If the matching key file is present ("Label,N" header, one row per label),
results are scored against it; otherwise clusters are just recorded. For
--variant a, if the key file is absent you must pass --num-groups and
--group-size explicitly, since the prompt needs to state them.
"""

import argparse
import csv
import json
import re
from collections import Counter
from math import comb
from pathlib import Path

from LLM_factory import chat_with_batch
from prompt_tasks import TASK3A_PLAIN_PROMPT, TASK3B_PLAIN_PROMPT
from task1 import format_table, load_dataset

DATA_FILES = {"a": "task3a_data.csv", "b": "task3b_data.csv"}
KEY_FILES = {"a": "task3a_key.csv", "b": "task3b_key.csv"}
OUTDIRS = {"a": "task3a_runs", "b": "task3b_runs"}


def load_key(path):
    """Return {label: N} from a task3{a,b}_key.csv ("Label,N" header), or None."""
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["Label"]: int(row["N"]) for row in reader if row.get("Label")}


def true_group_sizes(true_n):
    """Sorted (ascending N) list of true group sizes."""
    counts = Counter(true_n.values())
    return [count for _, count in sorted(counts.items())]


def extract_groups(text):
    """Return (groups, X, Y) from the last ```json block with a "groups" list.

    X / Y are the model's own estimates when present (variant b), else None.
    """
    for block in reversed(re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("groups"), list):
            groups = [list(g) for g in data["groups"]]
            return groups, data.get("X"), data.get("Y")
    return None, None, None


def adjusted_rand_index(true_labels, pred_labels):
    """Standard Adjusted Rand Index from a contingency table (no sklearn dependency).

    1.0 = predicted partition matches the true one up to relabeling,
    0.0 = chance-level agreement, negative = worse than chance.
    """
    n = len(true_labels)
    if n < 2:
        return 1.0

    contingency = Counter(zip(true_labels, pred_labels))
    row_sums = Counter(true_labels)
    col_sums = Counter(pred_labels)

    sum_comb_c = sum(comb(count, 2) for count in contingency.values())
    sum_comb_rows = sum(comb(count, 2) for count in row_sums.values())
    sum_comb_cols = sum(comb(count, 2) for count in col_sums.values())
    total_comb = comb(n, 2)

    expected = sum_comb_rows * sum_comb_cols / total_comb
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return (sum_comb_c - expected) / denom


def cluster_ari(groups, true_n):
    """Adjusted Rand Index between a predicted partition and the true one, clipped to [0, 1].

    1.0 = predicted clusters match the true ones up to relabeling, 0.0 = at
    or below chance-level agreement. Labels missing from `groups` (and
    labels the model hallucinated that aren't in `true_n`) are each treated
    as their own singleton cluster, so omissions are penalized rather than
    silently ignored.
    """
    all_labels = list(true_n)
    pred_of = {}
    for ci, members in enumerate(groups or []):
        for lab in members:
            if lab in true_n:
                pred_of.setdefault(lab, ci)  # first occurrence wins on duplicates
    next_id = len(groups or [])
    for lab in all_labels:
        if lab not in pred_of:
            pred_of[lab] = next_id
            next_id += 1

    true_labels = [true_n[lab] for lab in all_labels]
    pred_labels = [pred_of[lab] for lab in all_labels]
    return max(0.0, min(1.0, adjusted_rand_index(true_labels, pred_labels)))


def run_plain_batch(model, variant, labels, times, columns, n, num_groups=None, group_size=None, use_batch=False):
    if variant == "a":
        system_prompt = TASK3A_PLAIN_PROMPT.format(n=len(labels), num_groups=num_groups, group_size=group_size)
    else:
        system_prompt = TASK3B_PLAIN_PROMPT.format(n=len(labels))
    user_prompt = format_table(times, columns)
    print(f"[plain] querying {model} x{n} ({'batched' if use_batch else 'realtime'}) ...")
    replies = chat_with_batch(model, system_prompt, user_prompt, n=n, use_batch=use_batch)
    parsed = [extract_groups(r) if r else (None, None, None) for r in replies]
    return parsed, replies


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["a", "b"], required=True)
    p.add_argument("--data-file", default=None, dest="data_file")
    p.add_argument("--key-file", default=None, dest="key_file")
    p.add_argument("--num-groups", type=int, default=None, dest="num_groups",
                    help="variant a only; overrides/replaces the count derived from the key file")
    p.add_argument("--group-size", type=int, default=None, dest="group_size",
                    help="variant a only; overrides/replaces the count derived from the key file")
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

    true_n = load_key(key_file)

    num_groups, group_size = args.num_groups, args.group_size
    if args.variant == "a":
        if true_n is not None and (num_groups is None or group_size is None):
            sizes = true_group_sizes(true_n)
            if num_groups is None:
                num_groups = len(sizes)
            if group_size is None:
                if len(set(sizes)) != 1:
                    raise ValueError(f"task3a groups are not uniform in size ({sizes}); pass --group-size explicitly")
                group_size = sizes[0]
        if num_groups is None or group_size is None:
            raise ValueError("variant a needs --num-groups/--group-size (no key file to derive them from)")
        print(f"Disclosing to the model: {num_groups} groups of {group_size} trajectories each\n")

    base_outdir = Path(outdir_default)
    scale = args.scale

    # Plain mode: one prompt, all `scale` repeats fired via chat_with_batch.
    plain_parsed, plain_transcripts = run_plain_batch(
        args.model, args.variant, labels, times, columns, scale, num_groups, group_size,
        use_batch=args.batch,
    )

    trials = []
    for i in range(scale):
        if scale > 1:
            print(f"=== trial {i + 1}/{scale} ===")
        outdir = base_outdir / f"trial_{i + 1}" if scale > 1 else base_outdir
        outdir.mkdir(parents=True, exist_ok=True)
        results = {"model": args.model, "variant": args.variant}

        (groups, x_est, y_est), transcript = plain_parsed[i], plain_transcripts[i]
        (outdir / "transcript_plain.txt").write_text(transcript or "", encoding="utf-8")
        print(f"[plain] groups: {groups}  X={x_est}  Y={y_est}\n")
        results["plain"] = {"groups": groups, "X": x_est, "Y": y_est}
        if groups is not None and true_n is not None:
            results["plain_ari"] = cluster_ari(groups, true_n)

        results_path = outdir / f"results_{args.model}.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to {results_path}")
        trials.append(results)

    if scale > 1:
        summary = {"model": args.model, "variant": args.variant, "scale": scale, "trials": trials}
        for key in ("plain_ari",):
            scored = [t[key] for t in trials if key in t]
            if scored:
                summary[f"{key}_mean"] = sum(scored) / len(scored)

        summary_path = base_outdir / f"summary_{args.model}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
