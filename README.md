# Judge

## Anchored N-lobed Lorenz attractor

`anchored_n_lorenz.py` simulates a cylindrical-coordinate generalization of the
Lorenz system with N lobes ("wings") instead of the classic 2. It can run as
an interactive matplotlib window, or headless for scripted/batch use.

```
python anchored_n_lorenz.py --no-gui --N 5 --time 300 --save --png out.png --random --seed 30 --dt 0.2 --precision 8
```

- `--no-gui` — run once and exit instead of opening the interactive window.
- `--N 5` — simulate a 5-lobed attractor.
- `--time 300` — integrate for 300 time units.
- `--random --seed 30` — draw the initial condition from seed 30 (reproducible) instead of the default `--x0/--y0/--z0`.
- `--dt 0.2` — measurement (output) sampling interval; the integrator itself is adaptive, so this only controls how densely the trajectory is *recorded*, not its accuracy.
- `--precision 8` — significant digits used for x, y, z in the saved CSVs.
- `--save` — write the CSV/metadata files to `n_lorenz_saved_runs/`.
- `--png out.png` — render the 4-panel figure (xy / xz / yz projections + time series) to `out.png` instead of showing it.

## LLM ordering experiment

`experiment.py` simulates one trajectory per N in a range (default N=2..10),
shuffles and relabels them, and asks an LLM to recover the correct
ascending-N order purely from the data — in two conditions: `plain` (raw
series pasted into the prompt) and `code` (the LLM can write and execute
Python against the full-resolution files). See `prompt.py` for the exact
system prompts and `LLM_factory.py`/`config.json` for available models.

```
python experiment.py --mode both --model gpt-5.4 --N-max 10
```

- `--mode both` — run both the `plain` and `code` conditions (use `plain` or `code` to run just one).
- `--model gpt-5.4` — which model to query, must be listed in `config.json`.
- `--N-max 10` — use N=2 (default `--N-min`) through N=10.

Results (predicted order + scores) are written to `experiment_runs/results_<model>.json`,
and the full transcripts to `experiment_runs/transcript_plain.txt` / `transcript_code.txt`.

## Task 1 / 2 / 3: focused LLM judgment tasks

`task1.py`, `task2.py`, `task3.py` are three narrower variants of the same idea, each
simulating its own dataset and comparing an LLM's raw-data judgment (`plain`) against
its judgment when it can write and execute Python against the data (`code`). Prompts
live in `prompt_tasks.py` (one shared `BASE_SYSTEM_PROMPT`, plus a `PLAIN`/`CODE` prompt
pair per task); the dataset/prompting/code-execution plumbing they all share lives in
`task_common.py`. Every dataset is written as a single combined CSV: one `time` column,
then `<label>_x`, `<label>_y`, `<label>_z` per labeled trajectory, no quoting.

```
python task1.py --mode both --model gpt-5.4 --N-min 2 --N-max 10
python task2.py --mode both --model gpt-5.4 --N-min 2 --N-max 10 --delta 3
python task3.py --mode both --model gpt-5.4
```

- **task1 -- rank by N**: one trajectory per N in `[N_min, N_max]`, all from the SAME
  initial condition (`--seed`), so differences are due to N alone. The LLM recovers the
  ascending-N order of the shuffled labels. Scored by Kendall's tau / exact-position
  matches / mean absolute N error.
- **task2 -- spot the outlier(s)**: `N_max-N_min+1` trajectories share one lobe count
  `--n-base` (default: midpoint of `N_min`/`N_max`), each from a different seed; `--delta`
  (1-10) additional trajectories use a clearly different lobe count (at least `--gap`
  away). The LLM must name exactly the outlier labels. Scored by precision/recall/F1.
- **task3 -- group by complexity**: `--per-group` trajectories at each of three known
  lobe counts `--N-low`/`--N-mid`/`--N-high` (default 2/10/20). The LLM partitions the
  shuffled labels into "low"/"medium"/"high" groups. Scored by per-label accuracy.

Each script writes `experiment_runs/task<i>/dataset.csv`, `transcript_plain.txt` /
`transcript_code.txt`, and `results_<model>.json`, mirroring `experiment.py`'s layout.
