"""System prompts for the lobe-ordering (task1) and outlier-detection (task2)
experiments.

Each task has a PLAIN prompt (data pasted directly into the prompt). Remaining
{placeholders} are filled in by task1.py / task2.py via str.format(**kwargs).
"""

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

TASK3B_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis; N is an unknown positive integer that differs between trajectories. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. They form some unknown number of groups (call it X), each containing some unknown number of trajectories (call it Y): every trajectory in a group shares the same N (each started from a different initial condition, so their paths differ in phase and shape), and every group has a different N from every other group. Neither X nor Y is disclosed to you. Cluster the labels into groups, order the groups from least to most complex (lowest-N group first), and report your own estimates of X and Y.

Give your final answer as a single fenced json block containing all {n} labels, each exactly once, grouped as a list of lists ordered from least to most complex, plus your estimates of X and Y:

```json
{{"groups": [["<label>", "...", "<label>"], ...], "X": <int>, "Y": <int>}}
```

You may reason first, but the block must appear exactly once, at the end of your message. Do not restate the raw data in your response."""


# ---------------------------------------------------------------------------
# Task 4: estimate the lobe count N of every trajectory
# ---------------------------------------------------------------------------

TASK4_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis; N is an unknown positive integer that differs between trajectories. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. Estimate the lobe count N for each trajectory independently.

Give your final answer as a single fenced json block containing all {n} labels, each exactly once:

```json
{{"estimates": {{"<label>": <N>, "...": <N>}}}}
```

You may reason first, but the block must appear exactly once, at the end of your message. Do not restate the raw data in your response."""
