# Judge

## Task 1: order trajectories by number of lobes

`task1data_shuffled.csv` holds trajectories from a dynamical system: a
"time" column, then one column per labeled trajectory (a single letter),
each cell an `"(x, y)"` pair. Each trajectory wanders around N distinct
"lobes", where N differs between labels and is unknown to us -- the labels
were shuffled by whoever generated the data.

`task1.py` asks an LLM to recover the ascending-N order of the labels
purely from the data, pasted as a subsampled table directly into the
prompt.

Prompts live in `prompt_tasks.py`; models/providers are configured in
`config.json` and dispatched via `LLM_factory.py`.

```
python task1.py --model claude-opus-4-8
```

- `--model claude-opus-4-8` -- which model to query, must be listed in
  `config.json`.
- `--data-file task1data_shuffled.csv` -- dataset to read (default).
- `--points 120` -- rows shown per series in the prompt.

Each run writes `task1_runs/transcript_plain.txt` and
`task1_runs/results_<model>.json` (the model's predicted order). There is
no ground-truth key for this dataset, so results are not scored -- just
recorded.

## Task 2: spot the trajectory with a different lobe count

`task2data_shuffled.csv` is shaped like the task 1 dataset, but of its 20
labeled trajectories, 19 share the same number of lobes N (each started
from a different initial condition, so their paths differ) and exactly one
has a different N. The labels are shuffled so the outlier's position is
random.

`task2.py` asks an LLM to find the label of that one outlier:

```
python task2.py --model claude-opus-4-8
```

Same flags as `task1.py` (`--model`, `--data-file`, `--outdir`), plus
`--key-file task2_key.csv`. If that key file exists (columns
`output_label,is_outlier[,original_N]`, one row per label), results are
scored against it; otherwise the outlier is just recorded, unscored.

Each run writes `task2_runs/transcript_plain.txt` and
`task2_runs/results_<model>.json`.
