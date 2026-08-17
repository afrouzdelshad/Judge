"""System prompts for the lobe-ordering (task1) and outlier-detection (task2)
experiments.

Each task has a PLAIN prompt (data pasted directly into the prompt) and a
CODE prompt (extends it with CODE_PROTOCOL so the model can instead write
and execute Python against the CSV file). Remaining {placeholders} are
filled in by task1.py / task2.py via str.format(**kwargs).
"""

CODE_PROTOCOL = """

Instead of pasted data, you are given the path to a CSV file with this exact layout:
 - Row 1 is the header: "time", then one column per trajectory label (a single letter each), in no particular order.
 - Every following row is one time step, sorted by ascending time: a float time value, then one "(x, y)" cell per label column, e.g. "(1.23, -4.56)" -- literally formatted with parentheses and a comma, as a string. Standard CSV quoting is used, so each "(x, y)" cell is one quoted field; the comma inside it is not a column separator.

You can request Python code to be executed against this file.

File:
{data_file}

Protocol (up to {max_iters} exchanges total):
 - To run code, reply with ONLY a single fenced ```python code block. Its output is returned as your next message. `csv` plus `ast.literal_eval` per cell is a good way to load the file; `import numpy` yourself if you want it.
 - When confident, reply with ONLY the final fenced answer block described above -- no code block in that same message."""


# ---------------------------------------------------------------------------
# Task 1: order trajectories by ascending lobe count N
# ---------------------------------------------------------------------------

TASK1_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis; N is an unknown positive integer that differs between trajectories. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. Order the labels by ascending N (fewest lobes first, most last).

Give your final answer as a single fenced json block containing all {n} labels, each exactly once:

```json
{{"order": ["<lowest N>", "...", "<highest N>"]}}
```

You may reason first, but the block must appear exactly once, at the end of your message. Do not restate the raw data in your response."""

TASK1_CODE_PROMPT = TASK1_PLAIN_PROMPT + CODE_PROTOCOL


# ---------------------------------------------------------------------------
# Task 2: spot the one trajectory with a different lobe count
# ---------------------------------------------------------------------------

TASK2_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. {n_minus_1} of them share the same N (each started from a different initial condition, so their paths differ in phase and shape, but the lobe count is the same); exactly one trajectory has a different N. Find the label of that one outlier.

Give your final answer as a single fenced json block:

```json
{{"outlier": "<label>"}}
```

You may reason first, but the block must appear exactly once, at the end of your message. Do not restate the raw data in your response."""

TASK2_CODE_PROMPT = TASK2_PLAIN_PROMPT + CODE_PROTOCOL


# ---------------------------------------------------------------------------
# Task 2b: spot every trajectory that differs from the majority lobe count
# ---------------------------------------------------------------------------

TASK2B_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. Most of them share the same majority N (each started from a different initial condition, so their paths differ in phase and shape, but the lobe count is the same); the rest have an N that differs from that majority (and possibly from each other). Determine how many trajectories differ from the majority, and find the label of each one.

Give your final answer as a single fenced json block:

```json
{{"outliers": ["<label>", "..."]}}
```

List every outlier label exactly once, in any order; if you believe there are none, use an empty list. You may reason first, but the block must appear exactly once, at the end of your message. Do not restate the raw data in your response."""

TASK2B_CODE_PROMPT = TASK2B_PLAIN_PROMPT + CODE_PROTOCOL


# ---------------------------------------------------------------------------
# Task 3: cluster trajectories by inferred lobe count
# ---------------------------------------------------------------------------

TASK3A_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis; N is an unknown positive integer that differs between trajectories. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. They form exactly {num_groups} groups of {group_size} trajectories each: every trajectory in a group shares the same N (each started from a different initial condition, so their paths differ in phase and shape), and every group has a different N from every other group. Cluster the labels into their {num_groups} groups, and order the groups from least to most complex (lowest-N group first).

Give your final answer as a single fenced json block containing all {n} labels, each exactly once, grouped as a list of {num_groups} lists ordered from least to most complex:

```json
{{"groups": [["<label>", "...", "<label>"], ..., ["<label>", "...", "<label>"]]}}
```

You may reason first, but the block must appear exactly once, at the end of your message. Do not restate the raw data in your response."""

TASK3A_CODE_PROMPT = TASK3A_PLAIN_PROMPT + CODE_PROTOCOL

TASK3B_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis; N is an unknown positive integer that differs between trajectories. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. They form some unknown number of groups (call it X), each containing some unknown number of trajectories (call it Y): every trajectory in a group shares the same N (each started from a different initial condition, so their paths differ in phase and shape), and every group has a different N from every other group. Neither X nor Y is disclosed to you. Cluster the labels into groups, order the groups from least to most complex (lowest-N group first), and report your own estimates of X and Y.

Give your final answer as a single fenced json block containing all {n} labels, each exactly once, grouped as a list of lists ordered from least to most complex, plus your estimates of X and Y:

```json
{{"groups": [["<label>", "...", "<label>"], ...], "X": <int>, "Y": <int>}}
```

You may reason first, but the block must appear exactly once, at the end of your message. Do not restate the raw data in your response."""

TASK3B_CODE_PROMPT = TASK3B_PLAIN_PROMPT + CODE_PROTOCOL
