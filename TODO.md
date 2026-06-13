# TODO / deferred

Things parked for later, with enough context to pick up cold.

## Blocked on an idle GPU
- [ ] **Live smoke test of `run/protenix/run_benchmark_parallel.py`.** Validated by
  dry-run only (seed split, JSON merge, resume prefilter, command construction). Once
  a GPU frees up, run a tiny end-to-end check and confirm outputs land where eval
  expects:
  ```bash
  ~/Protenix/.venv/bin/python run/protenix/run_benchmark_parallel.py \
      --gpus <free_gpu> --seeds 101 --complexes 7st5 --n_step 20 \
      --dump_dir /tmp/protenix_smoketest
  # expect: /tmp/protenix_smoketest/7st5/seed_101/predictions/7st5_summary_confidence_sample_0.json
  ```
  Key things to confirm: merged-JSON path works, one checkpoint load, per-structure
  output dirs named by `name`, and a run config written under
  `/tmp/protenix_smoketest/configs/`.

## Optional / nice-to-have
- [ ] **Per-sample confidence head to fit large complexes (one diffusion sample at a
  time).** ConfidenceHead OOMs on large AbAg targets (≥~1,440 res — the 16 currently
  excluded: 8evd 7zau 8r4u 8xnh 7z2m 8oud 8fym 8en5 8pn0 8pyr 8j7t 8udz 8bf4 8dn7 8et0
  7z1x) because it runs a full folding trunk over the L×L pair batched across **all**
  `num_diffusion_samples` at once (`repeat_interleave` → Bm=5) in
  `transformers/models/esmfold2/modeling_esmfold2.py` (~L201 `pair =
  self._repeat_batch(z_base, num_diffusion_samples)`; trunk at ~L224). Even chunk_size 16
  OOMs at L≈1440 (~93 GB) — the per-sample base allocation is un-chunkable and the 5×
  batch is the multiplier. **Fix:** loop the confidence head over diffusion samples
  one-at-a-time (batch=1), concatenating per-sample outputs
  (plddt/pae/pde/ptm/iptm/pair_chains_iptm), instead of batching all 5. Cuts
  confidence-head peak ~5× → large complexes fit; numerically identical; keeps 5 samples
  + 50 seeds (honors the no-change-to-samples constraint). Trade-off: edits vendored
  site-packages `modeling_esmfold2.py` (fragile — re-apply after reinstall) and is
  slightly slower (serial over samples, but confidence is a small fraction of fold time).
  Then re-run the 16 excluded complexes (drop `--exclude_complexes`).
- [ ] **Switch example commands to `uv run` (user preference).** `esm` is a uv project
  (`pyproject.toml`+`uv.lock`); `Protenix` is not. Almost everything runs via the esm
  env — only `compute_msa_depths.py` imports `protenix`. Plan: update README + script
  docstring examples to `uv run --no-sync --project ~/esm python <script>` (the
  `--no-sync` guards the esm venv while jobs run); leave `compute_msa_depths.py` on
  `~/Protenix/.venv/bin/python` (Protenix isn't a uv project — converting it later is a
  separate task). Leave launcher *internal* subprocess plumbing (`ESM_PYTHON`/
  `PROTENIX_PYTHON`) as-is — those are the same envs uv would use.
- [ ] **Cross-run tracking at the eval stage.** If we want dashboards comparing run
  configs to DockQ outcomes, log `config + pass-rate` to wandb (or append a row to a
  `runs.csv`) from the eval/plot step — *not* the run step (runs emit no live metrics).
- [ ] **ESM launcher resume parity.** `run_foldbench_abag.py` loads the model before
  checking whether anything is left to do, so a fully-complete resume still pays one
  model load per GPU. Could prefilter (like the Protenix launcher does) and skip
  spawning a GPU whose work is already done.
- [ ] **ESM skip granularity.** Inner script skips a seed if *any* `*_sample_*.cif`
  exists; a seed interrupted mid-samples could be falsely treated as done. Consider
  requiring all `num_diffusion_samples` to be present.

## Cross-model consistency to decide deliberately (not bugs)
- [ ] **Seed conventions differ:** ESM uses `0–49` (strings, supports `baseline_seed`);
  Protenix uses `101–110` (ints). Fine as-is, but worth a deliberate choice/doc.
- [ ] **Diffusion steps differ:** ESM `--num_sampling_steps 16` vs Protenix
  `--n_step 200`. Partly model-inherent (fast model vs full Protenix), but be
  intentional about it for an apples-to-apples TTS comparison.
