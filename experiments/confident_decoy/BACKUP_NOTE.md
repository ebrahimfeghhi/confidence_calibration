# Confident-decoy experiments — code backup

Working directory is `/home/ebrahim/confident_decoy_experiments/` (47 GB, **not** a git repo).
This directory mirrors the parts worth version-controlling:

- `scripts/` — all analysis and pipeline code (82 files)
- `configs/` — per-run configs for the 2026-07-28 deployable-selection sweep
- `README.md`, `RESULTS.md` — the working-directory docs
- `PROJECT_README.md` — copy of `/home/ebrahim/confident_decoy_README.md`, the project entry point

**Not backed up here:** `artifacts/` (35 GB — OST score caches, template sets, prediction outputs) and
the Protenix predictions under `/home/ebrahim/Protenix/foldbench_abag/`. Those are regenerable from the
scripts plus the FoldBench ground truths, but regenerating the GPU runs is expensive. If the box is at
risk, `artifacts/*.json` (the scored results, a few hundred MB) is the part that cannot be cheaply
recomputed.

The narrative report lives at `reports/confident_decoy_REPORT.md` in this repo.

Sync: `rsync -a --exclude __pycache__ --exclude '*.pyc' <workdir>/scripts/ experiments/confident_decoy/scripts/`
