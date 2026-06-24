# archive/ — earlier 7txw steering & analysis experiments

Reference-only code from the exploration that led to the current ConforNet experiment.
Kept for provenance; **nothing here is imported by `confornet/`** (the portable core
extracted the few pieces it needed — `B1`/`B2` and `centroid()` now live in
`confornet/basins.py`).

The narrative tying these together — basins (G/B1/B2), the late-escape mechanism, the
gradient-steering and label-free repulsion results, and the pivot to ConforNets — is in
[`docs/steering_research_brief.md`](../docs/steering_research_brief.md).

- `steering/` — the gradient-steering lineage: `steer_lmz`, `steer_horizon`,
  `steer_coverage`, `steer_deviation`, `steer_sequential`, `steer_repel`, `steer_z0`,
  `steer_sweep`, `steer_converge`; `hotflip_mask.py`, `basin_w1_matrix.py`,
  `wg_k3_check.py`; the `decode_*`/`plot_*` helpers; run notes (`*.md`); and GPU-hold
  scripts. **`confornet_runner.sh` is the 4-GPU auto-grabber — do NOT use it** (it
  monopolizes shared GPUs; `confornet/confornet_one.sh` is the fair single-GPU version).
- `analysis/` — one-off analyses (PCA of reps/blocks, distogram-KL basin detectors, mask
  pattern / region-bias studies, pose clustering, recycle/stage profiling, etc.).

These scripts assume the original on-disk layout under `ESM_PREDICTIONS` (the 195 GB
`7txw_reps` and siblings) and were run with the `esm` venv; some paths may be stale.
