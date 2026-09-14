# Plan: Is there a mask pattern that biases 7txw toward the good antigen-orientation basin?

> Status: planned, not yet executed. Depends on the in-progress `7txw_tts2000` run.
> Scope decided: observational analysis **+ causal confirmation**; metastable-seed re-run deferred.

## Context

We've established for 7txw that landing in the correct antibody–antigen orientation
("good", A|C DockQ ≥ 0.5) is a recycling-driven commitment, the separating signal lives
in the **antigen (C) cross-terms** of the pairformer-z trunk rep, and the per-seed input
LM mask is **causal but exactly reproducible** (verified). The open question from
`tts-orientation-mask-null` — *do good seeds mask particular residues/regions?* — was
**underpowered at n=4 good seeds**, so no pattern was interpretable.

The in-progress `7txw_tts2000` run (seeds 60–2059, ~130 expected good) + the already-scored
`7txw_reps` set (seeds 0–59, 4 good) gives ~2060 seeds with reproducible masks and outcomes
— enough to test **with power** whether masking specific residues, or groups of residues,
makes a seed more likely to be good. Each seed's mask is ~10% of residues drawn i.i.d., so
under the null every residue is masked ~10% of the time independent of outcome; deviations
are the signal.

This reuses existing infrastructure rather than building new analysis from scratch.

## Approach

### 0. Inputs (outcomes + masks)
- **Score the run**: `eval/eval_ost_foldbench.py --pred_dir $ESM_PREDICTIONS/7txw_tts2000`
  → `dockq_ost_7txw_tts2000_all_samples.csv` (~10k structures). Combine with the existing
  `dockq_ost_7txw_reps_all_samples.csv` (seeds 0–59) for the full 0–2059 set.
- **Per-seed outcome**: confidence-selected A|C DockQ (best sample by `pair_iptm`, per
  `sample-selection-by-confidence`); label good = DockQ ≥ 0.5, bad = rest. (Robustness:
  also run good vs clearly-bad <0.23 excluding the 0.23–0.5 marginals.)
- **Per-seed masks**: regenerate `{seed: set of (asym, res) masked}` with
  `analyze_masks.generate_masks` for seeds 0–2059 (deterministic, CUDA; cheap — just seeded
  `torch.rand` draws). Build the binary matrix `M [n_seeds × 609]`.

### 1. Per-residue enrichment — *which residues*
Extend `compare_good_bad_masks.py` (which already computes f_good, f_bad, the Δ scatter, and
the within-good vs cross Jaccard permutation test) with the significance step it lacks now
that we have power:
- For each residue i: 2×2 table (masked/not × good/bad) → **Fisher's exact**, then **BH-FDR**
  across 609 residues. Effect size = risk ratio P(good|i masked)/P(good|i unmasked) and Δ=f_good−f_bad.
- Plot enrichment along the sequence, banded by chain (A 0–218, B 219–437, C 438–608) and
  with native-interface residues marked; scatter f_good vs f_bad.

### 2. Group / region enrichment — *groups of residues*
Scale `key_residue_masking.py` (already does per-group fraction-masked good-vs-bad +
label-shuffle p). Define groups via `analyze_masks.get_interface_ref` / `analyze_interface`
and the native cif:
- epitope (antigen C interface), paratope (antibody A/B interface), antibody framework,
  antigen non-epitope surface, and whole-chain A / B / C.
- Per group G: mean fraction of G masked, good vs bad, **permutation p**. Groups aggregate
  signal → more power than single residues, and are directly interpretable.

### 3. Multivariate predictive check — *does the mask predict the basin at all*
New small script: **L1-logistic regression** `M → good` (class-weighted), 5-fold **CV AUC**
with a **label-shuffle null**. If AUC > chance, the mask carries outcome information; inspect
nonzero coefficients for jointly-informative residues. Guards against (a) combinatorial
patterns the per-residue test misses and (b) false positives (CV + permutation).

### 4. Global confounds (cheap sanity)
Number of masked residues, and per-chain masked fraction, good vs bad (t-test). Rules out a
trivial "good seeds just mask more/less / mask the antigen more" explanation before
interpreting specific residues.

### 5. Connect to the orientation story
Cross-reference significant residues/groups against the **antigen-localization** result
(`pairformer-z-separates-good-seeds`: signal in A×C) and interface roles — do the
mask-enriched residues fall on the antigen epitope / paratope?

### Caveat from the reproducibility check: ~6% of seeds are metastable
Comparing the two existing 7txw runs (`esmfold2fast_tts` vs `7txw_reps`, identical config)
showed outcomes are **bimodal**: 47/50 seeds reproduce within |ΔDockQ|<0.1, 0 in 0.1–0.3, and
3/50 (seeds 19, 25, 39) switch basins *wholesale* (all 5 samples flip ~0.95↔~0.03). Masks are
identical across runs (seed 14 reproduces bit-for-bit; mask reproduction verified exact), so
these seeds sit on the basin **separatrix** where run-to-run nondeterminism (CUDA kernels over
17 recycling loops) flips the entire trunk commitment. Implication for this analysis:
- ~6% of seeds carry an **irreducible label flip** — their mask is genuinely basin-ambiguous, so
  no mask pattern can "explain" them. Treat as label noise.
- Robustness handling: flag metastable seeds whose 5 samples straddle the basins (bimodal
  per-seed DockQ), and report the per-residue/group results both with them included and excluded.
- **Deferred (per decision):** directly measuring the metastable fraction by re-folding boundary
  seeds (19/25/39) several times is a *later* experiment, not part of this build.

### Causal confirmation (IN SCOPE, runs after the observational pass)
Turn the top observational hit into causation by intervening on the mask:
- **Candidate set** = the FDR-significant residues / top-enriched group from steps 1–2.
- **Intervention**: fold a fresh batch of seeds (~200) under three conditions —
  (a) natural random mask (baseline good-rate), (b) candidate residues **force-masked** on top of
  the random mask, (c) a size-matched **control** set force-masked. Compare good-rates; (b) > (a) ≈ (c)
  ⇒ masking those residues causally raises the basin-hit rate.
- **Mechanism to build**: a way to inject a fixed mask into `compute_lm_hidden_states` instead of
  the random draw (the draw site is the `torch.rand < LM_MASK_PCT` in the LM path; add an optional
  `forced_mask` argument / monkeypatch in a new runner variant). ~200 seeds × 3 conditions × 5
  samples ≈ 3k folds; queues on the GPUs after the 2000-seed run, then score + compare.

## Files

- **Reuse as-is**: `analyze_masks.py` (`build_spi`, `build_lm_input_ids`, `generate_masks`,
  `all_residues`, `get_interface_ref`, `analyze_interface`); `compare_good_bad_masks.py`
  (`per_seed_dockq`, Jaccard + per-residue machinery); `key_residue_masking.py`
  (`group_mean_frac`, `perm_pvalue`, structural group defs); `plot/plot_scaling_ost.load_data`.
- **Modify**: parametrize the hardcoded `SEEDS = range(50)` and `DOCKQ_CSV =
  …esmfold2fast_tts…` in `analyze_masks.py` (accept a seed list + CSV path / complex) so the
  mask-analysis scripts can target `7txw_tts2000` + seeds 0–2059; generalize
  `compare_good_bad_masks.per_seed_dockq` to take a CSV path and complex id.
- **New**: `analysis/mask_pattern_powered.py` — orchestrator that builds M + outcomes, runs
  the per-residue Fisher/FDR test, the group permutation tests, and the L1-logistic CV, and
  writes one per-residue CSV, one group-summary CSV, and a summary figure
  (`analysis/figs/mask_pattern_7txw.*`). Plus a forced-mask runner variant for the causal step.

## Verification

- **Smoke test now** on seeds 0–59 (already scored): the extended scripts should run
  end-to-end at n=4 good and reproduce the prior null (no FDR-significant residues, group
  permutation p's ~uniform) — confirms wiring before the big data lands.
- **Null calibration**: under no effect, per-residue mean f_good ≈ f_bad ≈ 0.10; the
  label-shuffle / permutation p-values should be ~uniform; logistic CV AUC ≈ 0.5 on shuffled
  labels.
- **Full run** after `7txw_tts2000` is scored: report FDR-significant residues, significant
  groups with effect sizes, and CV AUC vs shuffle. A real pattern shows up as residues/groups
  with FDR<0.05 **and** the logistic model beating its shuffle null.
- **Causal step**: condition (b) good-rate significantly above (a)≈(c) confirms the residues
  are causal, not just correlated.
