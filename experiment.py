#!/usr/bin/env python3
"""
Can an LLM sort shuffled N-lobed Lorenz trajectories by N?
============================================================

For each N in [N_min, N_max] we simulate one anchored N-lobed Lorenz
trajectory (see anchored_n_lorenz.py), give each trajectory a random
single-letter label, and ask an LLM to recover the ascending-N order of the
labels purely from the data -- i.e. figure out which trajectory has the
fewest lobes, which has the most, and everything in between.

Two conditions:
  - "plain": the LLM only sees a subsampled (t, x, y, z) table pasted into
    the prompt and must reason about it directly.
  - "code":  the LLM is instead given file paths to the full-resolution CSVs
    and can iteratively write ```python code that this script executes
    locally, feeding stdout back to the model, before it commits to an
    answer.

Usage
-----
    python experiment.py --mode both --model claude-opus-4-8

Note: in "code" mode, model-written Python is exec'd locally with no
sandboxing. Only use this with trusted API providers on a machine you
control.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from anchored_n_lorenz import random_initial_condition, save_run, simulate
from LLM_factory import chat_with
from prompt import CODE_SYSTEM_PROMPT, PLAIN_SYSTEM_PROMPT

DEFAULT_OUTDIR = "experiment_runs"


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------

def generate_dataset(N_min, N_max, T, dt, precision, seed_base, outdir):
    """Simulate one trajectory per N, save it to disk, and return in-memory runs."""
    runs, files = {}, {}
    for N in range(N_min, N_max + 1):
        seed = seed_base + N
        x0, y0, z0 = random_initial_condition(seed)
        run = simulate(N, T, x0, y0, z0, seed, dt_out=dt)
        runs[N] = run
        saved = save_run(run, outdir=outdir, precision=precision)
        xyz_path = next(p for p in saved if str(p).endswith("_xyz_vs_time.csv"))
        files[N] = str(Path(xyz_path).resolve())
    return runs, files


def anonymize_files(label_to_N, files, outdir):
    """Copy each run's xyz CSV to a label-only filename.

    The originals are named like ..._N5_seed35_..._xyz_vs_time.csv, which
    would hand the answer to the "code" LLM for free just by reading the
    file path. The copies here carry only the (already-shuffled) label.
    """
    anon_dir = Path(outdir) / "anon"
    anon_dir.mkdir(parents=True, exist_ok=True)
    anon_paths = {}
    for label, N in label_to_N.items():
        dst = anon_dir / f"series_{label}.csv"
        dst.write_text(Path(files[N]).read_text(encoding="utf-8"), encoding="utf-8")
        anon_paths[label] = str(dst.resolve())
    return anon_paths


def assign_labels(Ns, shuffle_seed):
    labels = [chr(ord("A") + i) for i in range(len(Ns))]
    rng = np.random.default_rng(shuffle_seed)
    shuffled = list(Ns)
    rng.shuffle(shuffled)
    return dict(zip(labels, shuffled))  # label -> true N


def subsample(run, n_points):
    t, x, y, z = run["t"], run["x"], run["y"], run["z"]
    if n_points is None or n_points >= len(t):
        idx = np.arange(len(t))
    else:
        idx = np.unique(np.linspace(0, len(t) - 1, n_points).astype(int))
    return t[idx], x[idx], y[idx], z[idx]


def format_table(t, x, y, z, precision):
    lines = ["t,x,y,z"]
    for ti, xi, yi, zi in zip(t, x, y, z):
        lines.append(f"{ti:.4g},{xi:.{precision}g},{yi:.{precision}g},{zi:.{precision}g}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Prompt / response parsing
# --------------------------------------------------------------------------

def extract_json_order(text):
    matches = re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)
    for block in reversed(matches):  # prefer the last block (the final answer)
        try:
            data = json.loads(block)
            order = data.get("order")
            if isinstance(order, list) and all(isinstance(x, str) for x in order):
                return order
        except json.JSONDecodeError:
            continue
    return None


def extract_python_code(text):
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def execute_code(code):
    """Run model-generated code locally (unsandboxed) and capture stdout."""
    g = {"__builtins__": __builtins__, "np": np}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, g)
    except Exception:
        buf.write("\n[EXCEPTION]\n" + traceback.format_exc())
    out = buf.getvalue()
    if len(out) > 8000:
        out = out[:8000] + "\n...[truncated]"
    return out or "(no output)"


# --------------------------------------------------------------------------
# LLM runners
# --------------------------------------------------------------------------

def run_plain(model, label_to_N, runs, points, precision):
    n_min, n_max = min(label_to_N.values()), max(label_to_N.values())
    system_prompt = PLAIN_SYSTEM_PROMPT.format(n=len(label_to_N), n_min=n_min, n_max=n_max)

    blocks = []
    for label in sorted(label_to_N):
        t, x, y, z = subsample(runs[label_to_N[label]], points)
        blocks.append(f"### Series {label}\n{format_table(t, x, y, z, precision)}")
    user_prompt = "\n\n".join(blocks)

    reply = chat_with(model, system_prompt, user_prompt)
    return extract_json_order(reply), reply


def run_with_code(model, label_to_N, file_paths, max_iters):
    n_min, n_max = min(label_to_N.values()), max(label_to_N.values())
    listing = "\n".join(f"  {label}: {file_paths[label]}" for label in sorted(label_to_N))

    system_prompt = CODE_SYSTEM_PROMPT.format(
        n=len(label_to_N), n_min=n_min, n_max=n_max,
        file_listing=listing, max_iters=max_iters,
    )

    transcript = "Begin your analysis. Remember the protocol above."
    for _ in range(max_iters):
        reply = chat_with(model, system_prompt, transcript)
        transcript += f"\n\n[Assistant]\n{reply}"

        order = extract_json_order(reply)
        if order is not None:
            return order, transcript

        code = extract_python_code(reply)
        if code is None:
            transcript += ("\n\n[System]\nNo python code block or final json answer "
                            "found. Provide one or the other.")
            continue

        output = execute_code(code)
        transcript += f"\n\n[System: execution output]\n{output}"

    return None, transcript


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_order(pred_order, true_order, label_to_N, N_min):
    if pred_order is None:
        return {"parsed": False, "valid": False}
    if sorted(pred_order) != sorted(true_order):
        return {"parsed": True, "valid": False}

    labels = sorted(true_order)
    true_pos = {lab: i for i, lab in enumerate(true_order)}
    pred_pos = {lab: i for i, lab in enumerate(pred_order)}

    tau, _ = kendalltau([true_pos[l] for l in labels], [pred_pos[l] for l in labels])
    pred_N_for_label = {l: N_min + pred_pos[l] for l in labels}

    return {
        "parsed": True,
        "valid": True,
        "exact_position_matches": sum(true_pos[l] == pred_pos[l] for l in labels),
        "n_labels": len(labels),
        "kendall_tau": float(tau),
        "mae_N": float(np.mean([abs(pred_N_for_label[l] - label_to_N[l]) for l in labels])),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N-min", type=int, default=2, dest="N_min")
    p.add_argument("--N-max", type=int, default=10, dest="N_max")
    p.add_argument("--time", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.2, dest="dt_out")
    p.add_argument("--precision", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=30, dest="seed_base")
    p.add_argument("--shuffle-seed", type=int, default=7, dest="shuffle_seed")
    p.add_argument("--points", type=int, default=120,
                    help="points shown per series in the 'plain' prompt")
    p.add_argument("--max-iters", type=int, default=6, dest="max_iters",
                    help="max code/observe exchanges in 'code' mode")
    p.add_argument("--model", type=str, default="claude-opus-4-8")
    p.add_argument("--mode", choices=["plain", "code", "both"], default="both")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Simulating N={args.N_min}..{args.N_max} (T={args.time}, dt={args.dt_out}) ...")
    runs, files = generate_dataset(args.N_min, args.N_max, args.time, args.dt_out,
                                    args.precision, args.seed_base, args.outdir)

    label_to_N = assign_labels(list(runs.keys()), args.shuffle_seed)
    file_paths = anonymize_files(label_to_N, files, args.outdir)
    true_order = sorted(label_to_N, key=lambda l: label_to_N[l])

    print(f"Labels -> true N: {label_to_N}")
    print(f"True ascending order: {true_order}\n")

    outdir = Path(args.outdir)
    results = {"label_to_N": label_to_N, "true_order": true_order, "model": args.model}

    if args.mode in ("plain", "both"):
        print(f"[plain] querying {args.model} ...")
        order, transcript = run_plain(args.model, label_to_N, runs, args.points, args.precision)
        score = score_order(order, true_order, label_to_N, args.N_min)
        (outdir / "transcript_plain.txt").write_text(transcript, encoding="utf-8")
        print(f"[plain] predicted order: {order}")
        print(f"[plain] score: {score}\n")
        results["plain"] = {"predicted_order": order, "score": score}

    if args.mode in ("code", "both"):
        print(f"[code]  querying {args.model} ...")
        order, transcript = run_with_code(args.model, label_to_N, file_paths, args.max_iters)
        score = score_order(order, true_order, label_to_N, args.N_min)
        (outdir / "transcript_code.txt").write_text(transcript, encoding="utf-8")
        print(f"[code]  predicted order: {order}")
        print(f"[code]  score: {score}\n")
        results["code"] = {"predicted_order": order, "score": score}

    results_path = outdir / f"results_{args.model}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Full results written to {results_path}")


if __name__ == "__main__":
    main()
