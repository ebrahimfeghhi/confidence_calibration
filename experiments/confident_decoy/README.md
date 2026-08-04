# Confident-Decoy Experiments

Searchable, reproducible config registry for the confident-decoy rescue project.
Companion to `../confident_decoy_FINDINGS.md` (interpretation) and `../confident_decoy_RUNS.md`
(human narrative log). **This folder is the machine-readable source of truth for each run.**

## Layout
- `configs/*.yaml` — one config per experiment. Fully self-describing + reproducible.
- `scripts/` — durable copies of the builder / driver / scorer scripts (scratch is ephemeral).
- `artifacts/` — durable copies of input JSONs (jobs, antigen indices, interface sets).
- `logs/` — per-stage run logs written by `run_experiment.py`.
- `run_experiment.py` — replays a config's pipeline.

## Usage
```bash
python3 run_experiment.py --list                          # list all experiments + status/result
python3 run_experiment.py configs/abonly_decoys.yaml --dry-run   # print the exact commands
python3 run_experiment.py configs/abonly_decoys.yaml             # run build -> generate -> score
python3 run_experiment.py configs/abonly_decoys.yaml --stage score
```
Search: `grep -rl "template_chains: antibody" configs/`, `grep -l "status: done" configs/*.yaml`, etc.

## Config schema (every field required unless noted)
```yaml
id:            <name>_<YYYY-MM-DD>          # unique, matches filename
created:       YYYY-MM-DD
status:        planned | running | done | aborted
motivation:    |                            # WHY we run this; what each outcome would mean
               free-text, multi-line
hypothesis:    one line
set:           decoys | goods
interfaces:    {file: artifacts/<set>.json, n: 15}
method:
  template_chains:     antibody | antigen | both | none   # which chains get a template
  free_chains:         which chains are left template-free (MSA only)
  fold_source:         native | predicted_isolation | predicted_freerun | none
  template_selection:  exact_native | best_plddt | sample_0 | ...
  fixed_or_varying:    fixed | varying | n/a
  derivation:          | how the template JSONs were produced
generation:
  seeds:        "101-150"                    # seed-match baselines for fair comparison
  n_sample: 5   n_step: 200   use_template: true
  model: protenix-v2   checkpoint: /home/ebrahim/checkpoint
  gpus: [0,1,2,3]
provenance:     base_json / source_templates / antigen_index (exact paths)
baselines:      {free: "0/15", agonly: "8/15", native50: "14/15", ...}
scoring:        selection_rule + metric + scored_from
workdir:        <scratch selftmpl dir>       # where templates/outputs/scores live
pipeline:       [ {name, cmd:[...], cwd?}, ... ]   # exact, ordered, replayable
result:         {metric, value: "X/15", per_interface, notes}   # filled when done
```

## Conventions
- **Rescued (decoys):** selection (top interface-pair_iptm pose) DockQ >= 0.23.
- **Preserved (goods):** same threshold on the free-selection-correct control set.
- Always re-tally `result.value` from `scored_from`; never a remembered number.
- Seeds default to **101-150** (50 seeds) to stay seed-matched with `free`, `agonly`, `native50`.
- When a run finishes: set `status: done`, fill `result`, and add a line to `../confident_decoy_RUNS.md`.

## Reproducibility caveat
Pipeline scripts currently hardcode the scratch `SC=...selftmpl` path (see `workdir`). They
reproduce the run within this environment. For cross-machine reuse, parameterize `SC` via the
config `workdir` (TODO).
