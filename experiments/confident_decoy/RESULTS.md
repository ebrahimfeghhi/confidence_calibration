> ## ⚠️ CORRECTION 2026-08-04 — THREE ARMS IN THIS FILE ARE VOID
>
> `topdockq` (1/14), `topiptm` (0/5) and `sposefv` (2/8) **never loaded their templates.** Their run
> logs show `Found 0 templates` on every chain-load (1601, 451 and 1205 respectively). They measured
> UNTEMPLATED runs; their numbers say nothing about the ideas they were built to test.
>
> **Conclusion #5 below — "all 'which single pose's fold' ideas FAIL; mix-and-match oracle is the
> ceiling" — rests entirely on those three arms and is therefore UNSUPPORTED.**
>
> Re-running the DockQ-donor idea with working templates (`dockqdonor`, per interface, canonical 26)
> gives **9/25 rescued**, not 1/14. See `configs/dockqdonor_decoys.yaml` and report §6.
>
> Verified unaffected (templates loaded on every chain): `oracle` (1550), `abonly` (850-1052),
> `agonly` (1450), `native50` (1700), Exp B (1700). Exp C is 13% null and its 2/15 should be treated
> as provisional. Audit: `scripts/validate_run.py --logs-only <logdir>`.
>
> Root cause not fully established; the shared, mutable `/home/ebrahim/common/release_date_cache.json`
> (each builder rewrites the whole dict, so a concurrent builder can drop another's entry and make the
> date filter reject the template) is the leading candidate. Guard added: `score_arm.py` now refuses
> to score an arm whose templates never loaded.

# Confident-Decoy Rescue — Results (session 2026-07-27/28)

Durable summary of all runs. Raw per-pose scores in `artifacts/scores/`; per-run configs in
`configs/`; provenance in `../confident_decoy_RUNS.md`; interpretation in `../confident_decoy_FINDINGS.md`.
Selection rule everywhere: **top interface-pair_iptm pose's DockQ ≥ 0.23**, seeds 101–150 (250 poses).

## Headline table — decoy rescue (15 confident-decoy interfaces)

| run | template | decoys rescued | goods preserved |
|---|---|---|---|
| free | none | **0/15** | 15/15 (control: free selection already works) |
| **abonly** | native **antibody only** | **5/15** | **15/15** ✅ |
| agonly | native **antigen only** | 8/15 | 15/15 |
| **oracle** | best-**lDDT-to-native** fold, **per chain** (mix-and-match) | **8/15** | *(goods run failed — bug)* |
| native50 | both chains **exact native** | 14/15 | 15/15 |
| Exp D | best-**pLDDT** free fold, per chain | 0/15 | — |
| **sposefv** | oracle, but Fab H+L forced from **same pose** | **2/8 scored** (≈ oracle−) | — |
| **topiptm** | folds from the **top-ipTM (decoy) pose** | **0/5 scored** | — |
| **topdockq** | folds from the **top-DockQ (best-docked) pose** | **1/14 scored** (only 8ux6_DF) | — |

## Per-interface rescue sets
- **abonly (5):** 8aok_AB, 8av2_AC, 8q3j_AC, 8slb_AB, 8slb_AC  *(antibody-bottleneck)*
- **agonly (8):** 7z2m_AJ, 8bf4_AB, 8che_AE, 8che_BE, 8fbw_AC, 8ux6_DF, 8yxi_AC, 8yxi_BC  *(antigen-bottleneck)*
- **oracle (8):** 8bf4_AB, 8che_AE, 8che_BE, 8fas_BC, 8fbw_AC, 8q3j_AC, 8ux6_DF, 8yxi_BC
- **native50 (14):** all except 8fas_BC (the one exact-native can't fix)
- **sposefv (2/8):** 8yxi_AC, 8yxi_BC   *(stopped early — tracking ~= oracle with margin noise)*
- **topiptm (0/5):** none   *(killed — decoy fold reinforces decoy)*
- **topdockq (1/14):** 8ux6_DF   *(7z2m unscored; final ≈ 1/15)*

## Key results & conclusions

1. **abonly & agonly rescue DISJOINT sets (0 overlap).** Each confident decoy is bottlenecked on
   ONE chain's fold: antigen for 8, antibody for 5, both for 1 (8the_AB), unrescuable-even-by-native
   for 1 (8fas_BC). Union of singles = 13; native-both = 14; the 15th (8fas) no template fixes.
   → **abonly is SAFE (15/15 goods preserved).**

2. **oracle = 8/15 brackets the wall (Exp D 0/15 ⟶ oracle 8/15 ⟶ native 14/15).**
   - **Antigen side = SELECTION-limited:** the near-native fold exists in the pool; pLDDT can't pick
     it (rank-corr with fold accuracy ≈ **+0.01**). Oracle (cheating on selection) gets 6/8 antigen cases.
   - **Antibody side = GENERATION-limited:** even the model's best paratope fold is too imperfect
     (paratope lDDT ~0.86, down to 0.63); oracle gets only 1/5 antibody cases.

3. **pLDDT never picks the oracle fold (0/34 chains).** Harmless for antibodies only because their
   pool is tight (spread 0.055); for antigens the pool is wide (0.121) and the good fold is often a
   **rare outlier** — so pLDDT AND structural consensus both miss it. No deployable antigen-fold
   selector found (consensus rank-corr +0.19, still insufficient).

4. **Mechanism — ipTM depends on interface FOLD QUALITY, not just paratope/epitope identity.**
   A good fold RAISES the good pose's ipTM; only an EXACT fold also CRASHES the decoy's ipTM
   (making the wrong dock geometrically impossible). Deployable folds do the "raise good" half but
   not the "crash decoy" half (oracle raised good-pose ipTM 5/9, crashed decoy 1/9).

5. **"Which single pose's fold" all fail — mix-and-match oracle is the ceiling:**
   - **topiptm (decoy's own fold):** 0/5, and leaves good-pose ipTM UNCHANGED while shifting
     generation toward the decoy → *reinforces the decoy*.
   - **topdockq (best-docked pose's fold):** 1/14 — the best-*docked* pose does NOT carry the best
     *folds* (fold/dock decoupled), so it lost 7 of oracle's 8 (even 8bf4 with a 0.723-DockQ template → 0.003).
   - **sposefv (same-pose Fv):** ≈ oracle or marginally below; same-pose H+L is NOT better than
     mix-and-match (fold-level paratope Δ −0.006). The pairing worry was, if anything, inverted.

6. **Through-line: "confident decoys all the way down."** The model generates the right answer
   (both good docks AND near-native folds) but its confidence heads can't select it — ipTM mis-ranks
   the pose, pLDDT mis-ranks the fold. Generation was never the wall; **selection is, at every level.**
   The one open deployable lever: a better-than-pLDDT **antigen-fold selector** (the rare-outlier problem).

## Status / caveats
- **oracle_goods FAILED** (bug: a good-set native chain "AA0A" can't be written to legacy PDB in
  `prep_oracle_goods.py`; decoys unaffected). abonly_goods 15/15 already shows native-based
  templating is safe. Fix = write mmCIF / remap chain IDs.
- sposefv/topiptm/topdockq were stopped before full completion (conclusions were clear); counts
  above are over the interfaces actually scored.
- Runs still executing at session end: topdockq scheduler (7z2m only) + its incremental scorer.
