# Label-free repulsion steering — running experiment & handoff

Status doc for the reset-each-run K=3 repulsion runs. Read this to pick up the thread.

## Currently running (4 jobs, one per GPU)
All are `analysis/grad_steer/steer_repel.py`, reset-each-run, K=3, 10 runs:

| GPU | lever | M (steps/run) | z0 handling |
|---|---|---|---|
| 4 | z0  | 5  | fresh `fixed_z0(seed=run)` each run (the draw = diversity) |
| 5 | z0  | 15 | fresh per run |
| 6 | s_lm | 5  | **fixed** `z0_base` (pure s_lm repulsion) |
| 7 | s_lm | 15 | **fixed** z0 |

- Logs: `/tmp/repel_{z0,slm}_M{5,15}_K3.log`
- Per-run CSVs: `results/csv_files/steer_repel_{z0,slm}_M{5,15}_K3.csv`
  (columns: run, basin, wg/wb1/wb2 = W1 to G/B1/B2 centroids, dev, repel)
- Saved 17-loop z per run: `~/esm/foldbench_predictions/7txw_z0_repel/{z0,slm}_M{5,15}_K3/run*/seed_0/reps/pairformer_z_loop16.npy`

## What the script now does (the agreed design)
Reset-each-run greedy label-free repulsion — the proven `steer_sequential.py` recipe, K=3 throughout:
1. **Seed anchors** with the baseline's K=3 distogram (= B1).
2. **Each run (10):** RESET the working copy (z0 → fresh draw; s_lm → frozen baseline + fixed z0),
   optimize M steps to **maximize** mean W1 distance to all anchors (`loss = -mean_a W1(P_K3, a)`),
   then append the converged **K=3** distogram to the anchor set. Run 17 loops once to
   classify/save/decode (eval only — never in the loss). Discard the copy.
3. Only the **anchor set** carries across runs (RESET, not carry-forward — bounds off-manifold drift).

Key fixes vs the earlier (deleted) version: it was a single-leaf *walk* with a **K=1 loss + 17-loop
anchors** — wrong on two counts (carry-forward drifts off-manifold; K=1-vs-17-loop mixes a
maturity gap into the distance). Now: reset-each-run + **K=3 loss + K=3 anchors**.

## What to check when they finish
- **Basin trajectory** (CSV `basin` column): should march **B1 → B2 → G** as anchors accumulate.
- **`dev` bounded** (no cumulative drift — the payoff of resetting each run).
- **Decode → DockQ** (the real verdict): `decode_steered.py --steer_dir=.../7txw_z0_repel/<tag> --glob='run*'`
  then `eval_ost_foldbench.py`. Success = some run reaches G (DockQ ≳ 0.5, ideally ~0.9), matching
  `steer_sequential` (DockQ 0.85).
- **s_lm caveat:** fixed z0 + no nudge ⇒ every s_lm run starts byte-identical; diversity is *only*
  the growing anchor set (+ CUDA). If the s_lm runs collapse to one basin, add a small s_lm nudge
  (the "insurance" `steer_sequential` used).

## Why this design (context)
- `steer_sequential.py` (pre-existing) **discovered G unsupervised** via label-free repulsion:
  DockQ 0.85, B1→B2→G. It uses reset-each-run, fixed z0, s_lm nudge, K=3-consistent anchors.
- Supervised steering already reaches G (z0 0.92 at K=1 & K=3; s_lm 0.88–0.93 across K), so reaching
  G is possible; **this experiment tests whether *label-free* repulsion finds it** for z0 vs s_lm.
- Mechanism notes: z0 is an *early, transient* lever (gradient decays ~a^K, so K small); s_lm is
  *persistent* (re-injected every loop). See `docs/steering_gradients.md`.
- Full procedure + rationale: `/home/ebrahim/.claude/plans/1-compute-lm-hidden-states-runs-esmc-magical-grove.md`

## Open / TODO
- "10 runs" = one chain per config. For robustness, run multiple independent chains.
- Rename output dir `7txw_z0_repel` → lever-neutral (now holds z0 *and* slm).
- After runs: decode all 4 → DockQ, plot the per-run DockQ + basin trajectories.
- Old leftover `7txw_z0_repel/slm_M5_K1` is from a killed wrong-design run — can delete.
