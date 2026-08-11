"""System prompts for the N-lobed Lorenz ordering experiment (see experiment.py)."""

PLAIN_SYSTEM_PROMPT = """You are analyzing trajectories from a family of chaotic dynamical systems \
called "anchored N-lobed Lorenz attractors." Each member of the family is indexed by an \
integer N, and its trajectory wanders chaotically between N distinct lobes ("wings") \
arranged around a central axis -- generalizing the classic 2-lobe Lorenz butterfly to N \
wings. You are NOT given the governing equations or any formula; you must infer N purely \
from the shape and statistics of the raw (t, x, y, z) data itself.

You will be given {n} trajectories, each labeled with a single letter. Each \
trajectory was generated with a DIFFERENT integer number of lobes N, using \
every integer from {n_min} to {n_max} exactly once. The labels are given in \
no particular order.

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

CODE_SYSTEM_PROMPT = PLAIN_SYSTEM_PROMPT + """

Instead of the data being pasted in, you are given file paths to the \
full-resolution CSVs (columns: time,x,y,z) for each labeled trajectory, and \
you can request Python code to be executed against them.

Files:
{file_listing}

Protocol (repeat as many exchanges as you need, up to {max_iters} total):
 - To run code, reply with ONLY a single fenced ```python code block. It \
will be executed with numpy available as `np`, and whatever it prints will \
be shown to you as the next message. `np.loadtxt(path, delimiter=",", \
skiprows=1)` is a good way to load a file.
 - When confident, reply with ONLY the final fenced ```json answer block \
described above -- no code block in that same message."""
