"""System prompts for the task1/task2/task3 N-lobed Lorenz experiments.

Each task has a PLAIN prompt (data pasted directly into the prompt) and a
CODE prompt (the LLM instead gets a file path and can iteratively write and
execute Python against it -- see task_common.run_with_code). Both are built
by extending BASE_SYSTEM_PROMPT, the "overall" system prompt shared by every
task, with task-specific instructions and, for the code variant, the shared
CODE_PROTOCOL addendum. Task modules fill in the remaining {placeholders}
with str.format(**kwargs) before calling LLM_factory.chat_with.
"""

BASE_SYSTEM_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory \
wanders around N distinct "lobes" arranged around a central axis, where N is a positive \
integer that differs between trajectories. You are NOT given the governing equations or \
any formula; you must infer properties of each trajectory purely from the shape and \
statistics of the raw (t, x, y, z) data itself."""

CODE_PROTOCOL = """

Instead of the data being pasted in, you are given the path to a single CSV file \
holding every labeled trajectory side by side: one shared "time" column, then \
three columns per label named "<label>_x", "<label>_y", "<label>_z". You can \
request Python code to be executed against it.

File:
{data_file}

Protocol (repeat as many exchanges as you need, up to {max_iters} total):
 - To run code, reply with ONLY a single fenced ```python code block. It \
will be executed with numpy available as `np` (and pandas as `pd` if installed), \
and whatever it prints will be shown to you as the next message. \
`np.genfromtxt(path, delimiter=",", names=True)` is a good way to load the \
file and access columns by name, e.g. `data["A_x"]`.
 - When confident, reply with ONLY the final fenced answer block described \
above -- no code block in that same message."""


# ---------------------------------------------------------------------------
# Task 1: rank (N_max - N_min + 1) trajectories, one per N, by ascending N.
# All trajectories share the SAME initial condition, so differences are due
# to N alone.
# ---------------------------------------------------------------------------

TASK1_PLAIN_PROMPT = BASE_SYSTEM_PROMPT + """

You will be given {n} trajectories, each labeled with a single letter. Every \
one was generated from the SAME initial condition (so anything you see \
differing between them is due to N alone, not to different starting points), \
but with a DIFFERENT integer number of lobes N, using every integer from \
{n_min} to {n_max} exactly once. The labels are given in no particular order.

Your task: figure out the correct ascending order of the labels by their \
number of lobes N (so the first label in your answer used N={n_min}, the \
last used N={n_max}).

When ready, give your FINAL answer as a single fenced json block of exactly \
this form (and nothing else in that block):

```json
{{"order": ["<label with N={n_min}>", "...", "<label with N={n_max}>"]}}
```

You may reason in plain text before the block, but the block must appear \
exactly once, at the end of your message."""

TASK1_CODE_PROMPT = TASK1_PLAIN_PROMPT + CODE_PROTOCOL


# ---------------------------------------------------------------------------
# Task 2: spot the outlier(s). A majority of trajectories share (approximately)
# the same N; a small number of "delta" trajectories have a clearly different N.
# ---------------------------------------------------------------------------

TASK2_PLAIN_PROMPT = BASE_SYSTEM_PROMPT + """

You will be given {n_total} trajectories, each labeled with a single letter, \
in no particular order. {n_majority} of them share the SAME number of lobes \
N (each from a different random initial condition, so they will not look \
identical, but they share the same lobe count). The remaining {n_outliers} \
trajectories were generated with a CLEARLY DIFFERENT number of lobes N.

Your task: identify exactly the {n_outliers} label(s) that do NOT belong to \
the majority group.

When ready, give your FINAL answer as a single fenced json block of exactly \
this form (and nothing else in that block):

```json
{{"outliers": ["<label>", "..."]}}
```

List exactly {n_outliers} label(s), in any order. You may reason in plain \
text before the block, but the block must appear exactly once, at the end \
of your message."""

TASK2_CODE_PROMPT = TASK2_PLAIN_PROMPT + CODE_PROTOCOL


# ---------------------------------------------------------------------------
# Task 3: group trajectories into low/medium/high complexity (lobe count).
# Exactly `per_group` trajectories at each of three known candidate N values.
# ---------------------------------------------------------------------------

TASK3_PLAIN_PROMPT = BASE_SYSTEM_PROMPT + """

You will be given {n_total} trajectories, each labeled with a single letter, \
in no particular order. They come from exactly three complexity levels by \
number of lobes N -- "low" (N={n_low}), "medium" (N={n_mid}) and "high" \
(N={n_high}) -- with {per_group} trajectories at each level. You are not \
told which label belongs to which level.

Your task: partition the {n_total} labels into the three groups by \
complexity level.

When ready, give your FINAL answer as a single fenced json block of exactly \
this form (and nothing else in that block):

```json
{{"groups": {{"low": ["<label>", "..."], "medium": ["<label>", "..."], "high": ["<label>", "..."]}}}}
```

Each group must contain exactly {per_group} labels, and every label must \
appear in exactly one group. You may reason in plain text before the block, \
but the block must appear exactly once, at the end of your message."""

TASK3_CODE_PROMPT = TASK3_PLAIN_PROMPT + CODE_PROTOCOL
