"""Shared dataset / prompting / code-execution machinery for task1.py, task2.py, task3.py.

All three tasks follow the same shape: simulate a handful of labeled
anchored-N-lobed-Lorenz trajectories, show them to an LLM either as pasted
text ("plain") or as a single combined CSV the model can write and execute
Python against ("code"), and score the parsed final answer. This module
holds everything that does NOT vary between tasks; task-specific prompts
live in prompt_tasks.py and task-specific scoring/CLI logic lives in each
task_i.py.
"""

import contextlib
import io
import json
import re
import traceback
from pathlib import Path

import numpy as np

from anchored_n_lorenz import T_PRECISION, random_initial_condition, simulate
from LLM_factory import chat_with

DEFAULT_OUTDIR = "experiment_runs"


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------

def simulate_series(N, seed, T, dt_out):
    """Simulate one N-lobe trajectory from the initial condition drawn from `seed`."""
    x0, y0, z0 = random_initial_condition(seed)
    return simulate(N, T, x0, y0, z0, seed, dt_out=dt_out)


def write_combined_csv(path, label_runs, precision):
    """Write one CSV holding every labeled trajectory side by side.

    Columns: time, <label>_x, <label>_y, <label>_z, ... (sorted by label).
    Every run must share the same T/dt_out -- true automatically whenever
    every simulate_series() call in a task used the same T and dt_out,
    regardless of N or seed, so there is one shared time grid. No quoting
    is needed since every field is a plain number.
    """
    labels = sorted(label_runs)
    t = label_runs[labels[0]]["t"]
    v_fmt = f"%.{precision}g"

    cols = [t]
    header = ["time"]
    for label in labels:
        run = label_runs[label]
        cols += [run["x"], run["y"], run["z"]]
        header += [f"{label}_x", f"{label}_y", f"{label}_z"]

    fmt = [f"%.{T_PRECISION}g"] + [v_fmt] * (len(cols) - 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack(cols), fmt=fmt, delimiter=",",
               header=",".join(header), comments="")
    return path


def subsample_index(n_available, n_points):
    if n_points is None or n_points >= n_available:
        return np.arange(n_available)
    return np.unique(np.linspace(0, n_available - 1, n_points).astype(int))


def format_plain_table(label_runs, n_points, precision):
    """Wide, quote-free text table of all labeled series for the 'plain' prompt."""
    labels = sorted(label_runs)
    t = label_runs[labels[0]]["t"]
    idx = subsample_index(len(t), n_points)

    header = "t," + ",".join(f"{lab}_x,{lab}_y,{lab}_z" for lab in labels)
    lines = [header]
    for i in idx:
        row = [f"{t[i]:.4g}"]
        for lab in labels:
            run = label_runs[lab]
            row += [f"{run['x'][i]:.{precision}g}", f"{run['y'][i]:.{precision}g}",
                    f"{run['z'][i]:.{precision}g}"]
        lines.append(",".join(row))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

def extract_json_block(text, required_key):
    """Return the parsed dict from the last ```json block that has `required_key`."""
    matches = re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL)
    for block in reversed(matches):  # prefer the last block (the final answer)
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and required_key in data:
            return data
    return None


def extract_python_code(text):
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def execute_code(code):
    """Run model-generated code locally (unsandboxed) and capture stdout.

    Note: exec'd with no sandboxing. Only use "code" mode with trusted API
    providers on a machine you control.
    """
    g = {"__builtins__": __builtins__, "np": np}
    try:
        import pandas as pd
        g["pd"] = pd
    except ImportError:
        pass
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
# LLM runners (identical control flow for all three tasks; only the prompt
# text and the answer key differ)
# --------------------------------------------------------------------------

def run_plain(model, system_prompt, user_prompt, answer_key):
    reply = chat_with(model, system_prompt, user_prompt)
    return extract_json_block(reply, answer_key), reply


def run_with_code(model, system_prompt, max_iters, answer_key):
    transcript = "Begin your analysis. Remember the protocol above."
    for _ in range(max_iters):
        reply = chat_with(model, system_prompt, transcript)
        transcript += f"\n\n[Assistant]\n{reply}"

        answer = extract_json_block(reply, answer_key)
        if answer is not None:
            return answer, transcript

        code = extract_python_code(reply)
        if code is None:
            transcript += ("\n\n[System]\nNo python code block or final json answer "
                            "found. Provide one or the other.")
            continue

        output = execute_code(code)
        transcript += f"\n\n[System: execution output]\n{output}"

    return None, transcript


# --------------------------------------------------------------------------
# Top-level driver -- shared by task1.py / task2.py / task3.py so each of
# them only has to define its dataset and its scoring function.
# --------------------------------------------------------------------------

def run_experiment(model, mode, outdir, label_runs, plain_prompt, code_prompt,
                    prompt_kwargs, answer_key, score_fn, points, precision, max_iters):
    """Write the dataset, query the LLM in the requested mode(s), score, and save.

    `plain_prompt` / `code_prompt` are prompt_tasks.py templates; they get
    `.format(**prompt_kwargs, data_file=..., max_iters=...)` applied (extra
    keys that a given template doesn't use are simply ignored by str.format
    only if passed as **kwargs -- here every value is available to both).
    `score_fn(value)` scores the parsed `answer_key` field (None if missing).
    Returns the results dict, which is also written to results_<model>.json.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_file = write_combined_csv(outdir / "dataset.csv", label_runs, precision)
    print(f"Combined dataset written to {data_file}\n")

    fmt_kwargs = dict(prompt_kwargs, data_file=str(data_file.resolve()), max_iters=max_iters)
    results = {"model": model}

    if mode in ("plain", "both"):
        print(f"[plain] querying {model} ...")
        system_prompt = plain_prompt.format(**fmt_kwargs)
        user_prompt = format_plain_table(label_runs, points, precision)
        answer, transcript = run_plain(model, system_prompt, user_prompt, answer_key)
        value = answer.get(answer_key) if answer else None
        score = score_fn(value)
        (outdir / "transcript_plain.txt").write_text(transcript, encoding="utf-8")
        print(f"[plain] answer: {value}")
        print(f"[plain] score: {score}\n")
        results["plain"] = {"answer": value, "score": score}

    if mode in ("code", "both"):
        print(f"[code]  querying {model} ...")
        system_prompt = code_prompt.format(**fmt_kwargs)
        answer, transcript = run_with_code(model, system_prompt, max_iters, answer_key)
        value = answer.get(answer_key) if answer else None
        score = score_fn(value)
        (outdir / "transcript_code.txt").write_text(transcript, encoding="utf-8")
        print(f"[code]  answer: {value}")
        print(f"[code]  score: {score}\n")
        results["code"] = {"answer": value, "score": score}

    results_path = outdir / f"results_{model}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Full results written to {results_path}")
    return results
