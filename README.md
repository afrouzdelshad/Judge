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
