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

## Layout

```
paths.py            # single source of truth for every absolute path
run/
  esm/              # ESMFold2-Fast inference (needs the esm venv)
    run_foldbench_abag.py            # fold all complexes for a set of seeds
    run_foldbench_abag_parallel.py   # fan the above out across GPUs
  protenix/         # Protenix-v2 inference (needs the Protenix venv + checkpoint)
    run_benchmark_parallel.py        # 113-structure benchmark, seeds split across GPUs
eval/               # model-agnostic
    eval_ost_foldbench.py            # OST/DockQ for every sample -> CSV
    compute_avg_abag_iptm.py         # add avg_abag_iptm / new_ranking_score cols
    compute_msa_depths.py            # MSA depths from JSONs (needs Protenix venv)
plot/
    plot_scaling_ost.py              # TTS scaling curve + per-interface barplot
    plot_tts_scatter.py              # mean DockQ (N=1) vs best (N=50)
    plot_tts_slopegraph.py           # per-interface pass rate N=1 vs N=50
    plot_msa_depths.py               # MSA depth histogram
    plot_interface_atoms.py          # interface size histogram
analysis/
    analyze_masks_7yxu.py            # mask-pattern case study (needs esm venv)
results/            # default output dir for plots (git-ignored)
```

## Which environment runs what

There is no single venv — each stage uses the environment that has its model /
tools installed. Scripts find `paths.py` by walking up from their own location,
so you can invoke them from anywhere; just use the right interpreter.

| Stage | Interpreter / env | Why |
|-------|-------------------|-----|
| `run/esm/*` | `~/esm/.venv/bin/python` (`paths.ESM_PYTHON`) | imports the `esm` package |
| `run/protenix/run_benchmark_parallel.py` | `~/Protenix/.venv/bin/python` + `~/checkpoint` | calls Protenix `runner/inference.py` |
| `eval/eval_ost_foldbench.py` | any python with `tqdm`; needs the `foldbench` conda env's **OST** binary (`paths.OST_BIN`) | runs `ost compare-structures` |
| `eval/compute_msa_depths.py` | `~/Protenix/.venv/bin/python` | imports `protenix.data` |
| `eval/compute_avg_abag_iptm.py`, `plot/*` | any python with numpy/matplotlib | pure post-processing |
| `analysis/analyze_masks_7yxu.py` | `~/esm/.venv/bin/python` | imports the `esm` package |

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

## Overriding paths

Every constant in `paths.py` reads an env var of the same name first, e.g.:

```bash
FOLDBENCH_DIR=/data/FoldBench RESULTS_DIR=/tmp/plots \
    python plot/plot_scaling_ost.py --csv my_dockq.csv
```

Run `python paths.py` to print all resolved paths and whether each exists.
