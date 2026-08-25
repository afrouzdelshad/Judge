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

Structure your response as two fenced blocks, in this order: a ```reasoning block containing your reasoning process, followed by a ```json block containing your final answer with all {n} labels, each exactly once. Each block must appear exactly once, with the json block last. Do not restate the raw data in your response.

```reasoning
<your reasoning process>
```

```json
{{"order": ["<lowest N>", "...", "<highest N>"]}}
```"""


# ---------------------------------------------------------------------------
# Task 1B: order trajectories by ascending chaos level
# ---------------------------------------------------------------------------

TASK1B_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory exhibits a different degree of chaotic behavior. You are given only the raw ((t, x, y)) data and must infer the level of chaos for each trajectory.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. Order the labels in ascending order of chaos level (least chaotic first, most last).

Structure your response as two fenced blocks, in this order: a ```reasoning block containing your reasoning process, followed by a ```json block containing your final answer with all {n} labels, each exactly once. Each block must appear exactly once, with the json block last. Do not restate the raw data in your response.

```reasoning
<your reasoning process>
```

```json
{{"order": ["<least chaotic>", "...", "<most chaotic>"]}}
```"""


# ---------------------------------------------------------------------------
# Task 2: spot the one trajectory with a different lobe count
# ---------------------------------------------------------------------------

TASK2_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. {n_minus_1} of them share the same N (each started from a different initial condition, so their paths differ in phase and shape, but the lobe count is the same); exactly one trajectory has a different N. Find the label of that one outlier.

Structure your response as two fenced blocks, in this order: a ```reasoning block containing your reasoning process, followed by a ```json block containing your final answer. Each block must appear exactly once, with the json block last. Do not restate the raw data in your response.

```reasoning
<your reasoning process>
```

```json
{{"outlier": "<label>"}}
```"""


# ---------------------------------------------------------------------------
# Task 2b: spot every trajectory that differs from the majority lobe count
# ---------------------------------------------------------------------------

TASK2B_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. Most of them share the same majority N (each started from a different initial condition, so their paths differ in phase and shape, but the lobe count is the same); the rest have an N that differs from that majority (and possibly from each other). Determine how many trajectories differ from the majority, and find the label of each one.

Structure your response as two fenced blocks, in this order: a ```reasoning block containing your reasoning process, followed by a ```json block containing your final answer. List every outlier label exactly once, in any order; if you believe there are none, use an empty list. Each block must appear exactly once, with the json block last. Do not restate the raw data in your response.

```reasoning
<your reasoning process>
```

```json
{{"outliers": ["<label>", "..."]}}
```"""


# ---------------------------------------------------------------------------
# Task 3: cluster trajectories by inferred lobe count
# ---------------------------------------------------------------------------

TASK3A_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis; N is an unknown positive integer that differs between trajectories. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. They form exactly {num_groups} groups of {group_size} trajectories each: every trajectory in a group shares the same N (each started from a different initial condition, so their paths differ in phase and shape), and every group has a different N from every other group. Cluster the labels into their {num_groups} groups, and order the groups from least to most complex (lowest-N group first).

Structure your response as two fenced blocks, in this order: a ```reasoning block containing your reasoning process, followed by a ```json block containing your final answer with all {n} labels, each exactly once, grouped as a list of {num_groups} lists ordered from least to most complex. Each block must appear exactly once, with the json block last. Do not restate the raw data in your response.

```reasoning
<your reasoning process>
```

```json
{{"groups": [["<label>", "...", "<label>"], ..., ["<label>", "...", "<label>"]]}}
```"""

TASK3B_PLAIN_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory orbits N distinct "lobes" around a central axis; N is an unknown positive integer that differs between trajectories. You are given only the raw (t, x, y) data and must infer N for each trajectory from its shape and statistics.

You will be given {n} trajectories, each labeled with a single letter, in no particular order. They form some unknown number of groups (call it X), each containing some unknown number of trajectories (call it Y): every trajectory in a group shares the same N (each started from a different initial condition, so their paths differ in phase and shape), and every group has a different N from every other group. Neither X nor Y is disclosed to you. Cluster the labels into groups, order the groups from least to most complex (lowest-N group first), and report your own estimates of X and Y.

Structure your response as two fenced blocks, in this order: a ```reasoning block containing your reasoning process, followed by a ```json block containing your final answer with all {n} labels, each exactly once, grouped as a list of lists ordered from least to most complex, plus your estimates of X and Y. Each block must appear exactly once, with the json block last. Do not restate the raw data in your response.

```reasoning
<your reasoning process>
```

```json
{{"groups": [["<label>", "...", "<label>"], ...], "X": <int>, "Y": <int>}}
```"""


# ---------------------------------------------------------------------------
# Task 4: evaluate pairwise complexity/chaos claims about two trajectories
# ---------------------------------------------------------------------------

TASK4_PLAIN_PROMPT = """You are analyzing two trajectories, labeled A and B, from a dynamical system. Each trajectory orbits some number of distinct "lobes" around a central axis and may exhibit chaotic behavior to some degree; both properties are unknown and must be inferred from the raw (t, x, y) data.

Complexity means the number of lobes a trajectory visits; chaos means its dynamical instability (sensitivity to initial conditions). The two properties are distinct and need not covary.

Act as a scientific evaluator. Consider the following six candidate claims about the two trajectories:

1. Series A is more complex than Series B.
2. Series A is more chaotic than Series B.
3. Series B is more complex than Series A.
4. Series B is more chaotic than Series A.
5. Series A and Series B are equally complex.
6. Series A and Series B are equally chaotic.

Select every claim that you consider supported by the data. Claims 1/3/5 are mutually exclusive with each other (exactly one should hold for complexity), and claims 2/4/6 are mutually exclusive with each other (exactly one should hold for chaos).

Structure your response as two fenced blocks, in this order: a ```reasoning block containing your reasoning process, followed by a ```json block containing your final answer. Each block must appear exactly once, with the json block last. Do not restate the raw data in your response.

```reasoning
<your reasoning process>
```

```json
{{"claims": [<claim number>, "..."]}}
```"""
