# tts_abag

Benchmarking pipeline for **test-time scaling (TTS)** on the FoldBench
antibody–antigen task, across two structure-prediction models: **ESMFold2-Fast**
and **Protenix-v2**.

The pipeline is: **run** a model over many seeds → **eval** every diffusion
sample with OST/DockQ → **plot** the scaling curves and per-interface results.
The eval/plot/analysis code is model-agnostic — it reads a generic
`<complex>/seed_N/predictions/` layout, so it works on either model's output.

> **This repo holds scripts only.** Predictions, ground truths, checkpoints and
> input JSONs live in the original model repos / data dirs. All of those
> locations are centralized in [`paths.py`](paths.py) — edit that one file (or
> set the matching env vars) if anything moves.

> **Steering experiment:** the active work is label-free **distogram-repulsion steering**
> in [`steer/`](steer/README.md) — push ESMFold2 out of confident *wrong* antibody–antigen
> orientations without using the ground truth. Superseded steering/analysis experiments
> (including the earlier ConforNet line) are archived under `archive/`.

## Layout

```
paths.py            # single source of truth for every absolute path
run/
  esm/              # ESMFold2-Fast inference (needs the esm venv)
    run_foldbench_abag.py            # fold all complexes for a set of seeds
    run_foldbench_abag_parallel.py   # fan the above out across GPUs
    run_7txw_reps.py                 # capture baseline ESMC rep for a complex (steering prereq)
    run_7txw_loop_diffusion.py       # per-loop diffusion + load_complex helper (used by steer/core)
  protenix/         # Protenix-v2 inference (needs the Protenix venv + checkpoint)
    run_benchmark_parallel.py        # 113-structure benchmark, seeds split across GPUs
eval/               # model-agnostic
    eval_ost_foldbench.py            # OST/DockQ + confidence scores per sample -> CSV
plot/
    plot_scaling_ost.py              # TTS scaling curve + per-interface barplot
steer/              # label-free distogram-repulsion steering  (see steer/README.md)
    core/           # engine: forward_grad.py (differentiable trunk) + basins.py
    experiments/    # repel_proven.py (the method), decode_dockq.py (steered -> CIF -> DockQ)
results/            # default output dir for plots (git-ignored)
archive/            # superseded run/eval/plot/steer one-offs (nothing imports it)
```

## Which environment runs what

There is no single venv — each stage uses the environment that has its model /
tools installed. Scripts find `paths.py` by walking up from their own location,
so you can invoke them from anywhere; just use the right interpreter.

| Stage | Interpreter / env | Why |
|-------|-------------------|-----|
| `run/esm/*` | `~/esm/.venv/bin/python` (`paths.ESM_PYTHON`) | imports the `esm` package |
| `run/protenix/run_benchmark_parallel.py` | `~/Protenix/.venv/bin/python` + `~/checkpoint` | calls Protenix `runner/inference.py` |
| `eval/eval_ost_foldbench.py` | python with `tqdm` + `biotite` (else falls back to a `PROTENIX_PYTHON` subprocess); needs the `foldbench` conda env's **OST** binary (`paths.OST_BIN`) | runs `ost compare-structures` |
| `plot/*` | any python with numpy/matplotlib | pure post-processing |
| `steer/experiments/*` | `~/esm/.venv/bin/python` | imports the steering engine (`steer/core`) + `esm` |

## End-to-end example (ESMFold2 TTS)

```bash
cd ~/tts_abag

# 1) RUN — 50 seeds across 3 GPUs, 10% mask + 30% dropout (TTS settings)
~/esm/.venv/bin/python run/esm/run_foldbench_abag_parallel.py \
    --run_name esmfold2fast_tts --gpus 0 1 2 \
    --seeds $(seq 0 49) --lm_mask_pct 0.1 --lm_dropout 0.3
# writes to paths.ESM_PREDICTIONS/esmfold2fast_tts/<complex>/seed_N/predictions/

# 2) EVAL — DockQ for every sample (needs OST on PATH / via paths.OST_BIN)
python eval/eval_ost_foldbench.py \
    --pred_dir ~/esm/foldbench_predictions/esmfold2fast_tts \
    --out_csv  ~/esm/foldbench_predictions/dockq_ost_esmfold2fast_tts_all_samples.csv

# 3) PLOT — scaling curve + the per-interface barplot
python plot/plot_scaling_ost.py \
    --csv ~/esm/foldbench_predictions/dockq_ost_esmfold2fast_tts_all_samples.csv \
    --out results/tts_esmfold2fast_tts.png
# -> results/tts_esmfold2fast_tts.png  and  results/tts_esmfold2fast_tts_barplot.png
```

For Protenix, replace step 1 with the parallel launcher (same seeds-axis design as
ESM — each GPU runs all structures for its share of seeds, one checkpoint load per
GPU, resume-aware):

```bash
~/Protenix/.venv/bin/python run/protenix/run_benchmark_parallel.py \
    --gpus 0 1 3 4 --seeds $(seq 101 110)
# add --dry_run to preview per-GPU seeds, structure counts, and commands first
```

Then point `eval` at `paths.PROTENIX_PREDICTIONS`. Steps 2–3 are identical.

## Run configs

Every run records its exact parameters. Both launchers call `save_run_config`
(`runlog.py`), writing one JSON snapshot per invocation to
`<run_dir>/configs/<timestamp>__<uuid>.json` — params + timestamp, host, pid, and
the `tts_abag` git commit. The unique filename means concurrent GPU processes never
clobber each other (the launcher logs the run once; inner workers get
`--no_config_log`). So any prediction dir is self-describing; no external tracker
needed. (If you later want cross-run dashboards, the natural place is the eval stage,
logging config + DockQ pass-rate once results exist.)

## Overriding paths

Every constant in `paths.py` reads an env var of the same name first, e.g.:

```bash
FOLDBENCH_DIR=/data/FoldBench RESULTS_DIR=/tmp/plots \
    python plot/plot_scaling_ost.py --csv my_dockq.csv
```

Run `python paths.py` to print all resolved paths and whether each exists.
