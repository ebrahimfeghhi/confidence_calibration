# ESMFold2-Fast TTS benchmark (`esmfold2fast_tts`) — canonical reference

The main test-time-scaling (TTS) benchmark pool for antibody–antigen structure prediction, and the
data behind the confident-decoy analysis. **Read this first before re-deriving pool state.**

Last verified: 2026-07-07.

---

## 1. What the run is (the settings that actually generated it)

| field | value | source |
|---|---|---|
| model | `biohub/ESMFold2-Fast` | config.json |
| **lm_mask_pct** | **0.1** | every `configs/*.json` |
| **lm_dropout** | **0.3** | every `configs/*.json` |
| num_loops | 16 | config.json |
| num_sampling_steps | 16 | config.json |
| num_diffusion_samples | 5 (per seed) | config.json |
| chunk_size / fused | 32 / on | config.json |

> ⚠️ **The whole pool is MASKED (mask0.1/dropout0.3), NOT mask0/dropout0.** A prior memory note
> claimed mask0/drop0 — that was WRONG; it confused the per-complex `baseline_seed/` reference fold
> (the *only* mask0/drop0 fold) with the real TTS seeds. Verified from the run's own config logs +
> seed-dir mtimes. The mask0/drop0 data elsewhere in the project is the separate RandOpt
> weight-perturbation screen (`randopt_*`), which used mask0/drop0 on purpose to isolate weights.

## 2. Seed depth (folds on disk)

- **Original run:** seeds 0–49, folded 2026-06-14, all 96 complexes.
- **Deepening:** `steer/experiments/deepen27.sh` folded seeds 50–1023 (2026-07-06) for the **27
  severe** confident-decoy complexes (`gap>0.30`), same mask0.1/dropout0.3 settings, skip-existing.
  So those are at ~1024; the rest stay at 50.
- **Folds live at:** `~/esm/foldbench_predictions/esmfold2fast_tts/<cx>/seed_N/predictions/`
  (plus `baseline_seed/` = the single mask0/drop0 clean reference).

**Folds ≠ scores.** A structure on disk is not a DockQ number until scored. Check both at once:

```
python steer/experiments/tts_seed_status.py         # severe set: folds vs scored per complex
python steer/experiments/tts_seed_status.py --all   # all complexes
```

## 3. Score CSVs — which one to use

**Canonical:** `results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv`
— fixed `--dockq` pipeline, has `dockq_status`, consolidates all depth. Resume-safe (append).

Superseded / partial (don't build new analysis on these):

| file | coverage | note |
|---|---|---|
| `..._deep.csv` | all seeds, all cx | **USE THIS** |
| `..._verify.csv` | deep but ~360–762 max | pre-deepen27 snapshot; folded into `_deep` |
| `..._v2_all.csv` | 50 seeds only | has `dockq_status`; the barplot/figures used it |
| `..._all_samples.csv` | 50 seeds, no status col | oldest |
| `..._contam_v2.csv` | 2 cx | contamination spot-check |

**Scoring is fixed for the OST bug:** `eval/eval_ost_foldbench.py` drops `--rigid-scores`/`--tm-score`
(they crash on some structures and used to discard a valid DockQ as 0). Never re-add them.

To score more / resume:
```
python eval/eval_ost_foldbench.py \
  --pred_dir ~/esm/foldbench_predictions/esmfold2fast_tts \
  --out_csv results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv
```
It reads the out_csv, skips already-scored `(assembly,chain1,chain2,seed,sample)`, appends the rest.

## 4. The confident-decoy ("severe trap") set

- **Definition:** a good dock EXISTS in the pool (best DockQ high) but ipTM SELECTS a decoy
  (max-pair_iptm structure has DockQ < 0.23). "Minor slip" = ipTM picks a sub-best but still-good
  dock (gap ~0.05–0.2). "No basin" = no good dock exists at all (not a trap — a coverage failure).
- **Set:** 55 interfaces / 34 complexes → `results/csv_files/decoy_structural_analysis.csv`.
- **Selection tables:** `iptm_selection_gap_baseline.csv` (per-complex, 97), and
  `severe27_selection_by_interface.csv` (per-interface, 50).
- **Structural finding:** the decoy paratope is correct CDRs (~88% CDR) aimed at the WRONG antigen
  surface (58% dock a different surface). Not a CDR-vs-framework bias. Epitope selection is broken.
- **Recompute the selection contrast** (oracle vs ipTM-selected vs baseline) on any pool:
  `steer/experiments/randopt_select_analysis.py` (env vars `RANDOPT_CX/BASE/MINE`).

## 5. Gotchas / settled questions

- **8c5h, 8che:** *not* in the severe set, and correctly so. In the deep pool ipTM selects their
  GOOD dock (0.835 / 0.852); they only look severe in a small (~51-seed) pool. Pool-size-sensitive
  borderline, not a bug exclusion. The score masters are bug-clean (orig vs fixed differ in 0 rows).
- **8im0:** no reachable good dock (best ~0.06) → coverage failure, not a selection trap. Framework
  epitope. Correctly excluded.
- **ANSWERED 2026-07-07 — depth does NOT rescue selection.** At 1024 masked seeds, 51/55 trap
  interfaces survive (only 9fww, 9g2g resolve); mean ipTM-selected DockQ flat ~0.08 vs oracle ~0.6;
  more seeds can even hurt (8aok 0.61→0.15). Genuine ipTM failure, not undersampling. Analysis:
  `steer/experiments/severe_selection_at_depth.py`; see memory `confident-decoy-selection-depth-robust`.
  Next lever = a NON-confidence label-free selector (recycling/dynamics features) or transfer.

## 6. File map (quick jump)

- Fold config / provenance: `~/esm/foldbench_predictions/esmfold2fast_tts/configs/*.json`
- Deepen more seeds: `steer/experiments/deepen27.sh`
- Score: `eval/eval_ost_foldbench.py`
- Pool status: `steer/experiments/tts_seed_status.py`
- Selection contrast: `steer/experiments/randopt_select_analysis.py`
- Seed-scaling barplot: `results/tts_esmfold2fast_tts_v2_all_n1_2_5_10_20_30_50_barplot.png`
- DockQ-vs-ipTM figures: `steer/experiments/plot_dockq_vs_iptm.py` (grid PNG/PDF + pooled density),
  `steer/experiments/plot_dockq_iptm_slides.py` (3x3 slides). Fold check: `decoy_fold_quality.py`.
- **Interactive artifacts (pinned):** slideshow https://claude.ai/code/artifact/1dc08128-db4a-4380-ae45-6c8d54c0c956
  · pan/zoom grid https://claude.ai/code/artifact/204ad834-6346-4806-ac84-0218e6dbe758
