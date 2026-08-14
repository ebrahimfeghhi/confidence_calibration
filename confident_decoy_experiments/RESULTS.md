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

---

# 250-donor × 10-seed sweep on 4 more complexes (session 2026-08-11/12)

Design (`scripts/build_allpose_generic.py`): one job per free-run pose (seeds 101-150 × samples 0-4 =
250 donors), every protein chain templated from that single pose, 10 seeds per donor (own seed + 9
random per donor), `N_sample 5` → 2500 seeds / 12,500 poses per arm. Selection rule unchanged and
label-free: **the single highest interface-pair_iptm pose over all 12,500; report its DockQ.**

Scored with `scripts/score_allpose.py` (one OST call per pose, full native vs full model,
`--fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq`, interface matched BY NAME).
Per-arm artifacts: `artifacts/allpose*_scores.json`, `artifacts/allpose*_posetable.json` (full-precision
per-pose ipTM/DockQ), OST caches `artifacts/allpose*_ost/`. All arms verified `Found 0 templates` = 0.

| interface | free | **sweep** | oracle | n poses ≥0.6 | donors w/ acceptable | verdict |
|---|---|---|---|---|---|---|
| 8che_AE | 0.125 | **0.938** | 0.958 | 57 | 249/250 | rescued |
| 8che_BE | 0.087 | **0.924** | 0.957 | 53 | 247/250 | rescued |
| 9jbq_AC | 0.041 | **0.739** | 0.784 | 100 | 118/250 | rescued |
| 8av2_AC | 0.092 | **0.652** | 0.771 | 35 | 35/250 | rescued |
| 8fas_BC\* | 0.175 | 0.174 | 0.992 | 4072 | 215/250 | see \* |
| 8slb_AB | 0.012 | 0.021 | 0.789 | 42 | 207/250 | FAILS |
| 8slb_AC | 0.010 | 0.009 | 0.840 | 141 | 197/250 | FAILS |

**\* 8fas_BC is a DEGENERATE TARGET — do not count it as a selection failure.** The antigen is not a
folded domain: it is the circumsporozoite **NANP5 peptide, `NANPNANPNANPNANPNANP` — five identical
tandem repeats** (8fas title: "Ky230 Fab in complex with circumsporozoite protein NANP5 peptide";
GT chain C resolves only 11 of 20 residues). The Fab binds one NANP unit and **nothing distinguishes
which one**, so a pose shifted by 4 residues is chemically the same binding mode. Evidence that this is
what is happening: GT, the DockQ-0.992 pose and the DockQ-0.174 pose all contact the **same light-chain
residues (31, 33, 37, 96, 97, 99)**; only the peptide register differs (GT contacts peptide 5-9, the
0.992 pose contacts 9-13 = +4 = one repeat). OST aligns model to reference by sequence, and on identical
repeats that alignment can slide, so DockQ here is partly reporting register rather than binding mode.

**PROVEN: the DockQ values on this target are an ALIGNMENT ARTEFACT, not a pose error.**

Superposing the ipTM-selected pose's light chain alone onto the GT light chain (217 CA, RMSD 0.83 A,
label numbering) puts its peptide on the crystal's peptide **residue for residue, same numbering**:

```
GT residue:  4    5    6    7    8    9    10   11   12   13   14
distance:    0.6  0.4  0.3  0.3  0.3  0.5  0.5  0.5  0.6  0.8  1.2 A     (mean 0.55 A)
```

Yet OST reports `fnat 0.0`, `lrmsd 9.94`, DockQ **0.174** for exactly this pose (re-run from scratch,
reproduces the cache). The cause is visible in OST's own `aln` output:

```
>reference:C     -------PNANPNANPNA--
>model:C         NANPNANPNANPNANPNANP
```

The reference's resolved peptide (`PNANPNANPNA` = residues 4-14) has been aligned to model positions
**8-18** -- a **+4 slide, exactly one NANP unit**. OST therefore compares crystal residue 4 against
model residue 8, one repeat away (~10 A), so every native contact registers as missed.

**Consequences.** On this target DockQ scores whichever register OST's aligner happens to pick, so
*every* 8fas_BC DockQ is unreliable: the "oracle 0.992", the 4072 poses counted as >=0.6, the
corr(ipTM, DockQ) = -0.53, and the verdict that the no-MSA arm failed (0.166). None of those should be
quoted. The ipTM-selected pose is a near-perfect prediction that the metric mislabels; if anything,
8fas_BC is evidence that ipTM WORKED here.

Method note for anyone repeating this: use `label_seq_id`, not gemmi's auth numbering -- auth lumps
ligands/waters into chain A (480 residues numbered 1-656 vs 222 in label numbering) and pairing on it
produces a meaningless 13 A superposition that looks like a failed prediction.

So the honest reading is **4 rescued, 2 genuine failures (8slb_AB/AC, one complex), 1 degenerate**.

**8slb is a real epitope error, not a register artefact.** All 52 poses outranking the best good pose
recover **0.00** of the true epitope (max 0.11) and converge on one alternative surface (pairwise
Jaccard 0.48; antigen residues ~46-50/67-70/90-100 vs the true epitope at 76-77/100-103/125-127).
Every predicted metric prefers the wrong pose (pTM 0.824 vs 0.788/0.768; ipTM 0.807 vs 0.748/0.723;
`ranking_score` 0.810 vs 0.756/0.732) while every ground-truth metric rejects it (iLDDT 0.539 vs
0.775/0.869; QS 0.477 vs ~0.85; DockQ 0.009 vs 0.611). pLDDT is flat (89.4 vs 90.0/90.5) and there are
no clashes, so neither would filter it. The preference is **intrinsic to the model, not caused by
templating**: in the 755 untemplated poses the top-20 by ipTM also recover 0.00 of the epitope, even
though 98/755 free poses recover ≥50% of it.

**Donor DockQ is not what makes a good template.** corr(donor_dockq, out_max) = -0.05 (9jbq), +0.12
(8che_AE), +0.27 (8fas). The 9jbq winner's own DockQ was **0.041** (rank 169/250); 8av2's was 0.061
(rank 106/250). This follows from `multichain_mask` (`Protenix/protenix/model/modules/pairformer.py:1013,1062`)
zeroing every cross-chain template feature — a donor can only ever transmit intra-chain folds, never a
binding arrangement. The oracle-donor rule (`dockqdonor`) was therefore optimising a quantity the model
cannot see; it still beats the sweep on 8slb (0.734/0.887) by *suppressing* the decoy rather than
excluding it (decoy poses are still generated: 59/250, but top out at ipTM 0.508 vs 0.557 for correct).

Pending at time of writing: `allpose8fas10nomsa` scored (0.166 — no-MSA does not fix 8fas either),
`allpose8slb10nomsa` queued, and compute-matched free baselines `freedeep{8fas,9jbq,8av2,8slb}`
(2349 seeds each, 151-2499, extending the existing 0-150 to 2500 total, matching `freedeep8che`).
