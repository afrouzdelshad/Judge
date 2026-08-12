"""System prompts for the task1 lobe-ordering experiment.

TASK1_PLAIN_PROMPT is used when the data is pasted directly into the prompt;
TASK1_CODE_PROMPT extends it with CODE_PROTOCOL so the model can instead
write and execute Python against the CSV file. Both get the remaining
{placeholders} filled in by task1.py via str.format(**kwargs).
"""

BASE_SYSTEM_PROMPT = """You are analyzing trajectories from a dynamical system. Each trajectory \
wanders around N distinct "lobes" arranged around a central axis, where N is a positive \
integer that differs between trajectories. You are NOT given the governing equations or \
any formula; you must infer properties of each trajectory purely from the shape and \
statistics of the raw (t, x, y) data itself."""

CODE_PROTOCOL = """

Instead of the data being pasted in, you are given the path to a CSV file holding every \
labeled trajectory side by side: a "time" column, then one column per label (its letter), \
each cell holding an "(x, y)" pair as a string like "(1.23, -4.56)". You can request \
Python code to be executed against it.

File:
{data_file}

Protocol (repeat as many exchanges as you need, up to {max_iters} total):
 - To run code, reply with ONLY a single fenced ```python code block. It will be executed, \
and whatever it prints will be shown to you as the next message. Python's `csv` module \
plus `ast.literal_eval` on each cell is a good way to load the file; `import numpy` \
yourself if you want it.
 - When confident, reply with ONLY the final fenced answer block described above -- no \
code block in that same message."""


TASK1_PLAIN_PROMPT = BASE_SYSTEM_PROMPT + """

You will be given {n} trajectories, each labeled with a single letter, in no particular \
order. Each was generated with a different number of lobes N -- a higher N means more, \
tighter lobes arranged around the central axis.

Your task: order the labels by ascending N (fewest lobes first, most lobes last).

When ready, give your FINAL answer as a single fenced json block of exactly this form \
(and nothing else in that block):

```json
{{"order": ["<label with lowest N>", "...", "<label with highest N>"]}}
```

The list must contain all {n} labels, each exactly once. You may reason in plain text \
before the block, but the block must appear exactly once, at the end of your message."""

TASK1_CODE_PROMPT = TASK1_PLAIN_PROMPT + CODE_PROTOCOL
