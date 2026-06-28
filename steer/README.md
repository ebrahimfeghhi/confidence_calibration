# `steer/` — label-free distogram-repulsion steering

Test-time steering of ESMFold2-Fast to fix **antibody–antigen orientation** failures: the
model often folds both chains correctly but docks them in the wrong (and confident) relative
orientation. We push the model out of that wrong basin **without using the ground-truth
structure**, by repelling each prediction away from the ones we've already seen.

## The idea (sequential anchor accumulation)

Everything happens in **cross-distogram space** — the antibody×antigen block of the model's
predicted distance distribution, which *is* the relative orientation and is available from the
first forward pass (no labels needed).

For one target we do a sequence of "runs". Each run optimizes a per-residue leaf `s_lm` (a small
nudge to the baseline language-model rep) so that its cross-distogram is **far in Wasserstein-1
(`w1`) from every previous run's cross-distogram** — the accumulated *anchors*. Run 1 escapes the
baseline; run 2 escapes runs ≤1; etc. This is label-free: G/B1/B2 references are only used to
*score* what was found, never to drive the optimization.

Key knobs (`repel_proven.py`):
- `--K` — recycling horizon the loss acts at. **Sweet spot ~K=8–10** (deep but pre-commitment).
  Too early (K=1–2) the good basin hasn't formed; too late (K=16) it's already committed → B2.
- `--M` — gradient steps per run (proven recipe M=5–6).
- `--bptt N` — truncated backprop (grad through last N recycles only). Cheap, but **full backprop
  reaches *deep* G; truncated only reaches the rim** — use full backprop when you want real hits.
- `--multik 3,6,9` — apply the loss at several horizons at once (truncated), to avoid tuning K.
- `--target` — `7txw`, `8im0`, … (chain slices + rep paths live in `forward_grad.TARGETS`).

## Layout

```
steer/
├── core/            # the engine (stable import surface)
│   ├── forward_grad.py   # differentiable trunk, s_lm leaf, cross_distogram, w1, configure_target, TARGETS
│   └── basins.py         # load_centroids -> G/B1/B2 reference distograms (7txw)
├── data/            # portable bundle: centroids/, baseline reps  (gitignored, ~300 MB)
└── experiments/     # the method + curated diagnostics (+ co-located outputs, gitignored)
    ├── repel_proven.py        # THE method (sequential anchor repulsion)
    ├── decode_dockq.py        # steered s_lm -> 3D structure (CIF), for DockQ
    └── perloop_*/velocity_probe/trajectory_depth/basin_width/...  # diagnostics
```

## End-to-end workflow

All commands use the esm venv: `PY=/home/ebrahim/esm/.venv/bin/python`. Steering runs go in a
per-GPU tmux session; one trial of 10 runs is ~10–20 min (target-size dependent).

**1. Run ESMFold / capture the baseline rep** (prerequisite for a new target)
```bash
# full prediction (structures + confidence)
$PY run/esm/run_foldbench_abag.py --run_name <name> --complexes <pdb> --seeds 0..N
# baseline ESMC rep that steering nudges (no mask, no dropout)
$PY run/esm/capture_reps.py --run_name <pdb>_reps_baseline --complex <pdb> \
      --seeds 0 --lm_mask_pct 0 --lm_dropout 0
```

**2. Steer** (label-free; writes repel_*.csv, pairwise_*.csv, outcomes/leaves_*.npy beside the script)
```bash
cd steer/experiments
CUDA_VISIBLE_DEVICES=N $PY repel_proven.py --target <pdb> --K 6 --M 6 --lr 0.3 \
      --runs 10 --trials 1 --tag <pdb>_k6 --pairwise
```

**3. Decode steered leaves to structures** (inject steered `lm_z` -> model -> CIF)
```bash
CUDA_VISIBLE_DEVICES=N $PY decode_dockq.py leaves_<pdb>_k6.npy   # -> decoded_<pdb>_k6/*.cif
# ("sanity" instead of a leaves file decodes the baseline as a check)
```

**4. Compute DockQ** (OST vs native)
```bash
$PY eval/eval_ost_foldbench.py --pred_dir <dir of decoded CIFs> --out_csv results/csv_files/dockq_<pdb>_steered.csv
```

**5. Plot / analyze**
```bash
$PY plot/plot_scaling_ost.py --csv results/csv_files/<dockq>.csv --out results/<plot>.png
```

## Key findings (7txw)

- **There is a commitment window.** Real-G hits vs loss horizon K: 0,0 (K=1,2) → 3,0,2,3,4
  (K=3,4,6,8,10) → collapses at K=16. Both extremes fail: too early the basin isn't formed; too
  late it's committed and runs re-collapse into the confident decoy (B2). Sweet spot is *deep*
  (K=8–10), and the window is broad (K=3–10 all work) — so you don't need to nail K exactly.
- **Full backprop ≫ truncated for deep hits.** Truncated (`--bptt 1`, `--multik`) reaches G's rim
  (near-G) but rarely the core; full backprop lands real-G.
- **Repulsion works on `w1` at the loss horizon, but the outcome (K=17) can re-collapse** — the
  K-loss→outcome map is many-to-one over the B2 sink. That's *why* horizon matters.

## Adding a new target

1. Capture its baseline rep (step 1 above) → `ESM_PREDICTIONS/<pdb>_reps_baseline/<pdb>/seed_0`.
2. Add an entry to `forward_grad.TARGETS` (antibody `ab` / antigen `c` slices — **chain order is
   not assumable**, e.g. 7txw is antibody-first, 8im0 is antigen-first + single VHH) and the
   baseline path. Verify chain slices against the complex JSON.
3. Scoring on a new target has no G/B1/B2 centroids: `repel_proven.py` runs in `gt_mode`
   (`--target != 7txw`), scoring each run's **escape** = `w1` to the baseline basin, and saving
   `outcomes_*.npy` for later distance-to-native / DockQ scoring.

## Caveats

- **CUDA nondeterminism flips basins run-to-run** (dropout/recycling knife-edge). The same leaf can
  decode to different ipTM/structure across processes — so DockQ a *few* decodes per leaf, not one.
- Outputs are co-located with scripts and **gitignored**; only code + `SYNTHESIS.md` are tracked.
- `steer/data/` (centroids + baseline reps) is the portable bundle; on a fresh machine it must be
  present (the from-`REPS` fallback needs the full ~195 GB `7txw_reps`).
