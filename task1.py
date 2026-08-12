#!/usr/bin/env python3
"""
Task 1: rank same-initial-condition trajectories by N
========================================================

One trajectory is simulated per N in [N_min, N_max], all from the SAME
initial condition (same seed), so any difference the LLM sees is due to N
alone. Each trajectory gets a random single-letter label, and the LLM must
recover the ascending-N order of the labels -- either from a pasted table
("plain") or by writing Python against a combined CSV ("code").

Usage
-----
    python task1.py --mode both --model claude-opus-4-8 --N-min 2 --N-max 10

Note: in "code" mode, model-written Python is exec'd locally with no
sandboxing. Only use this with trusted API providers on a machine you
control.
"""

import argparse

import numpy as np
from scipy.stats import kendalltau

from prompt_tasks import TASK1_CODE_PROMPT, TASK1_PLAIN_PROMPT
from task_common import run_experiment, simulate_series

DEFAULT_OUTDIR = "experiment_runs/task1"


def build_dataset(N_min, N_max, seed, T, dt_out):
    """One run per N (same seed), shuffled onto random letter labels.

    `seed` drives both the shared initial condition and the label shuffle.
    """
    Ns = list(range(N_min, N_max + 1))
    np.random.default_rng(seed).shuffle(Ns)

    labels = [chr(ord("A") + i) for i in range(len(Ns))]
    label_to_N = dict(zip(labels, Ns))
    label_runs = {label: simulate_series(N, seed, T, dt_out) for label, N in label_to_N.items()}
    return label_to_N, label_runs


def score_order(pred_order, true_order, label_to_N, N_min):
    if pred_order is None or sorted(pred_order) != sorted(true_order):
        return {"parsed": pred_order is not None, "valid": False}

    true_pos = {lab: i for i, lab in enumerate(true_order)}
    pred_pos = {lab: i for i, lab in enumerate(pred_order)}
    tau, _ = kendalltau([true_pos[l] for l in true_order], [pred_pos[l] for l in true_order])

    return {
        "parsed": True,
        "valid": True,
        "exact_position_matches": sum(true_pos[l] == pred_pos[l] for l in true_order),
        "kendall_tau": float(tau),
        "mae_N": float(np.mean([abs((N_min + pred_pos[l]) - label_to_N[l]) for l in true_order])),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N-min", type=int, default=2, dest="N_min")
    p.add_argument("--N-max", type=int, default=10, dest="N_max")
    p.add_argument("--seed", type=int, default=30,
                   help="single seed controlling both the shared initial condition and the label shuffle")
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

    print(f"Simulating N={args.N_min}..{args.N_max} (shared seed={args.seed}) ...")
    label_to_N, label_runs = build_dataset(args.N_min, args.N_max, args.seed, args.time, args.dt_out)
    true_order = sorted(label_to_N, key=lambda l: label_to_N[l])
    print(f"Labels -> true N: {label_to_N}")
    print(f"True ascending order: {true_order}\n")

    run_experiment(
        model=args.model, mode=args.mode, outdir=args.outdir, label_runs=label_runs,
        plain_prompt=TASK1_PLAIN_PROMPT, code_prompt=TASK1_CODE_PROMPT,
        prompt_kwargs=dict(n=len(label_to_N), n_min=args.N_min, n_max=args.N_max),
        answer_key="order",
        score_fn=lambda order: score_order(order, true_order, label_to_N, args.N_min),
        points=args.points, precision=args.precision, max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main()
