#!/usr/bin/env python3
"""
Task 2: spot the outlier trajectories among a same-N majority
================================================================

(N_max - N_min + 1) trajectories all share the SAME N ("n_base", default:
midpoint of N_min/N_max), each from a different initial condition. `delta`
additional trajectories are added with a clearly different N (at least
`gap` away from n_base). All are shuffled onto random letter labels, and
the LLM must name exactly the `delta` outlier labels -- either from a
pasted table ("plain") or by writing Python against a combined CSV
("code").

Usage
-----
    python task2.py --mode both --model claude-opus-4-8 --N-min 2 --N-max 10 --delta 3

Note: in "code" mode, model-written Python is exec'd locally with no
sandboxing. Only use this with trusted API providers on a machine you
control.
"""

import argparse

import numpy as np

from prompt_tasks import TASK2_CODE_PROMPT, TASK2_PLAIN_PROMPT
from task_common import run_experiment, simulate_series

DEFAULT_OUTDIR = "experiment_runs/task2"


def build_dataset(N_min, N_max, n_base, delta, gap, seed, T, dt_out):
    """n_majority runs at n_base + delta runs clearly away from it, shuffled onto labels.

    `seed` drives both the per-run initial conditions (run i gets seed + i)
    and the label shuffle.
    """
    n_majority = N_max - N_min + 1
    all_Ns = [n_base] * n_majority + [n_base + gap + i for i in range(delta)]

    order = list(range(len(all_Ns)))
    np.random.default_rng(seed).shuffle(order)

    labels = [chr(ord("A") + i) for i in range(len(all_Ns))]
    label_to_N, label_runs, true_outliers = {}, {}, []
    for label, i in zip(labels, order):
        N = all_Ns[i]
        label_to_N[label] = N
        label_runs[label] = simulate_series(N, seed + i, T, dt_out)
        if i >= n_majority:
            true_outliers.append(label)
    return label_to_N, label_runs, sorted(true_outliers), n_majority


def score_outliers(pred, true_outliers, all_labels):
    if pred is None or not isinstance(pred, list):
        return {"parsed": pred is not None, "valid": False}

    pred_set, true_set = set(pred) & set(all_labels), set(true_outliers)
    tp, fp, fn = len(pred_set & true_set), len(pred_set - true_set), len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "parsed": True,
        "valid": True,
        "exact_match": pred_set == true_set,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N-min", type=int, default=2, dest="N_min")
    p.add_argument("--N-max", type=int, default=10, dest="N_max")
    p.add_argument("--n-base", type=int, default=None, dest="n_base",
                   help="lobe count shared by the majority (default: midpoint of N_min/N_max)")
    p.add_argument("--delta", type=int, default=3, choices=range(1, 11),
                   help="number of outlier trajectories (1-10)")
    p.add_argument("--gap", type=int, default=None,
                   help="how much larger an outlier's N is than n_base (default: N_max-N_min+2)")
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
    n_base = args.n_base if args.n_base is not None else (args.N_min + args.N_max) // 2
    gap = args.gap if args.gap is not None else args.N_max - args.N_min + 2

    print(f"Simulating {args.N_max - args.N_min + 1} majority runs at N={n_base} "
          f"+ {args.delta} outlier(s) starting at N={n_base + gap} ...")
    label_to_N, label_runs, true_outliers, n_majority = build_dataset(
        args.N_min, args.N_max, n_base, args.delta, gap, args.seed, args.time, args.dt_out)
    print(f"Labels -> true N: {label_to_N}")
    print(f"True outliers: {true_outliers}\n")

    run_experiment(
        model=args.model, mode=args.mode, outdir=args.outdir, label_runs=label_runs,
        plain_prompt=TASK2_PLAIN_PROMPT, code_prompt=TASK2_CODE_PROMPT,
        prompt_kwargs=dict(n_total=len(label_to_N), n_majority=n_majority, n_outliers=args.delta),
        answer_key="outliers",
        score_fn=lambda outliers: score_outliers(outliers, true_outliers, sorted(label_to_N)),
        points=args.points, precision=args.precision, max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main()
