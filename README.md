# Judge

## Task 1: order trajectories by number of lobes

`task1data_shuffled.csv` holds trajectories from a dynamical system: a
"time" column, then one column per labeled trajectory (a single letter),
each cell an `"(x, y)"` pair. Each trajectory wanders around N distinct
"lobes", where N differs between labels and is unknown to us -- the labels
were shuffled by whoever generated the data.

`task1.py` asks an LLM to recover the ascending-N order of the labels
purely from the data, in two conditions:

- `plain` -- a subsampled table pasted directly into the prompt.
- `code` -- the LLM is given the CSV path and can write and execute Python
  against it.

Prompts live in `prompt_tasks.py`; models/providers are configured in
`config.json` and dispatched via `LLM_factory.py`.

```
python task1.py --mode both --model claude-opus-4-8
```

- `--mode both` -- run both the `plain` and `code` conditions (use `plain`
  or `code` to run just one).
- `--model claude-opus-4-8` -- which model to query, must be listed in
  `config.json`.
- `--data-file task1data_shuffled.csv` -- dataset to read (default).
- `--points 120` -- rows shown per series in the `plain` prompt.
- `--max-iters 6` -- max back-and-forth exchanges in `code` mode.

Each run writes `task1_runs/transcript_plain.txt` / `transcript_code.txt`
and `task1_runs/results_<model>.json` (the model's predicted order for each
mode). There is no ground-truth key for this dataset, so results are not
scored -- just recorded.

Note: in `code` mode, model-written Python is exec'd locally with no
sandboxing. Only use this with trusted API providers on a machine you
control.
