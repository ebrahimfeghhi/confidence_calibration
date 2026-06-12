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
