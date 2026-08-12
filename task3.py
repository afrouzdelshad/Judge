#!/usr/bin/env python3
"""
Task 3: group trajectories by complexity level (lobe count)
==============================================================

`per_group` trajectories are simulated at each of three N values -- N_low,
N_mid, N_high (default 2, 10, 20) -- each from a different initial
condition. All 3 * per_group trajectories are shuffled onto random letter
labels; the LLM is told the three candidate N values and must partition
the labels into "low"/"medium"/"high" groups -- either from a pasted
table ("plain") or by writing Python against a combined CSV ("code").

Usage
-----
    python task3.py --mode both --model claude-opus-4-8

Note: in "code" mode, model-written Python is exec'd locally with no
sandboxing. Only use this with trusted API providers on a machine you
control.
"""

import argparse

import numpy as np

from prompt_tasks import TASK3_CODE_PROMPT, TASK3_PLAIN_PROMPT
from task_common import run_experiment, simulate_series

DEFAULT_OUTDIR = "experiment_runs/task3"
LEVELS = ("low", "medium", "high")


def build_dataset(n_low, n_mid, n_high, per_group, seed, T, dt_out):
    """per_group runs at each of the three N levels, shuffled onto random letter labels.

    `seed` drives both the per-run initial conditions (run i gets seed + i)
    and the label shuffle.
    """
    all_Ns = [n_low] * per_group + [n_mid] * per_group + [n_high] * per_group
    all_levels = [LEVELS[0]] * per_group + [LEVELS[1]] * per_group + [LEVELS[2]] * per_group

    order = list(range(len(all_Ns)))
    np.random.default_rng(seed).shuffle(order)

    labels = [chr(ord("A") + i) for i in range(len(all_Ns))]
    label_to_N, label_to_level, label_runs = {}, {}, {}
    for label, i in zip(labels, order):
        N = all_Ns[i]
        label_to_N[label] = N
        label_to_level[label] = all_levels[i]
        label_runs[label] = simulate_series(N, seed + i, T, dt_out)
    return label_to_N, label_to_level, label_runs


def score_groups(pred, true_level, all_labels, per_group):
    if pred is None or not isinstance(pred, dict) or set(pred) != set(LEVELS):
        return {"parsed": pred is not None, "valid": False}

    pred_level = {label: level for level, labels in pred.items()
                  for label in (labels if isinstance(labels, list) else [])}
    if set(pred_level) != set(all_labels):
        return {"parsed": True, "valid": False}

    correct = sum(pred_level[l] == true_level[l] for l in all_labels)
    return {
        "parsed": True,
        "valid": True,
        "accuracy": correct / len(all_labels),
        "group_sizes_correct": all(len(v) == per_group for v in pred.values()),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N-low", type=int, default=2, dest="n_low")
    p.add_argument("--N-mid", type=int, default=10, dest="n_mid")
    p.add_argument("--N-high", type=int, default=20, dest="n_high")
    p.add_argument("--per-group", type=int, default=5, dest="per_group")
    p.add_argument("--seed", type=int, default=30,
                   help="single seed controlling both the per-run initial conditions and the label shuffle")
    p.add_argument("--time", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.2, dest="dt_out")
    p.add_argument("--precision", type=int, default=8)
    p.add_argument("--points", type=int, default=120,
                   help="points shown per series in the 'plain' prompt")
    p.add_argument("--max-iters", type=int, default=6, dest="max_iters")
    p.add_argument("--model", type=str, default="claude-opus-4-8")
    p.add_argument("--mode", choices=["plain", "code", "both"], default="both")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Simulating {args.per_group} runs each at N={args.n_low}/{args.n_mid}/{args.n_high} "
          f"(low/medium/high) ...")
    label_to_N, label_to_level, label_runs = build_dataset(
        args.n_low, args.n_mid, args.n_high, args.per_group, args.seed, args.time, args.dt_out)
    all_labels = sorted(label_to_N)
    print(f"Labels -> true N: {label_to_N}")
    print(f"True levels: {label_to_level}\n")

    run_experiment(
        model=args.model, mode=args.mode, outdir=args.outdir, label_runs=label_runs,
        plain_prompt=TASK3_PLAIN_PROMPT, code_prompt=TASK3_CODE_PROMPT,
        prompt_kwargs=dict(n_total=len(all_labels), n_low=args.n_low, n_mid=args.n_mid,
                            n_high=args.n_high, per_group=args.per_group),
        answer_key="groups",
        score_fn=lambda groups: score_groups(groups, label_to_level, all_labels, args.per_group),
        points=args.points, precision=args.precision, max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main()
