# Confident-Decoy Rescue — Key Findings

**Goal:** find a *deployable* method to rescue "confident decoys" — antibody–antigen
interfaces where Protenix's ipTM selects a structurally wrong pose (DockQ < 0.23) over
50 seeds, even though a correct pose exists in the pool. 15 decoy interfaces (12 unique
complexes; some complexes contribute 2 interfaces).

---

## 1. The bottleneck is SELECTION, not generation

- The **free complex runs already have 15/15 coverage** — a good pose (DockQ ≥ 0.23)
  exists in the 250-pose pool (50 seeds × 5 samples) for every decoy interface.
  Generation was never the wall.
- ipTM (the confidence head) ranks a **structurally wrong pose on top**. That mis-ranking
  is the entire failure.

## 2. Exact-native templates rescue — by killing the decoy's false ipTM

Feeding the *exact native* fold of each chain as a per-chain, interface-masked template
(templates carry only intra-chain fold; the cross-chain block is zeroed by
`multichain_mask`) rescues **14/15**.

- Mechanism (`mech.py`): native does **not** mainly change generation — it **collapses the
  decoy's inflated ipTM**. Free runs: the decoy out-confidences the good pose 15/15 (the
  pathology). Native flips it 15/15, driven by the decoy's ipTM crashing
  (e.g. 8bf4 decoy ipTM 0.857 → 0.345, good pose ~flat).
- Regimes: GEN (decoy no longer generated) 4/15; RANK (decoy kept but demoted) 11/15.
- **Not deployable** — requires the answer (native structure).

## 3. The error localizes to CDR-H3

Per-residue CA-lDDT of the best deployable fold vs native:

- Framework lDDT **0.94** (native-quality) vs **CDR-H3 0.77** — a +0.17 gap concentrated
  in one loop.
- **Scales with CDR-H3 length:** short loops (7–11 res) near-native (0.88–0.98); long loops
  (18–23 res) badly modeled (0.48–0.72). For long H3, even the oracle-best of 250 poses is
  ~0.78 → **generation-limited**, not selectable.
- **Global lDDT (~0.90–0.95) hides this** — framework is ~90% of the chain and drowns the
  loop. Region-restricted lDDT is what reveals it.
- **All 15 interfaces bind via CDR-H3** (16–67% of the heavy-chain paratope).

## 4. Deployable folds are NOT good enough as templates — a dose-response on accuracy

The number rescued scales with **how exact the template is**. This is the core result:

| Template (chains fed) | Fold source | Rescued | Run |
|---|---|---|---|
| Native **both** chains (Ab + Ag) | native (exact) | **14/15** | `native50` — ceiling; not deployable |
| Native **antigen only** (Ab free) | native (exact) | **8/15** | `agonly` — isolates antigen-side signal |
| Predicted **both** chains, isolation folds, fixed | predicted | **2/15** | `expb` —  antibody/antigen folded in isolation|
| Predicted **both** chains, free-run poses, diverse | predicted | **2/15** | `expc` — template derived from seed itself|
| Predicted **both** chains, single best-pLDDT, fixed | predicted | **0/15** | Exp D (`expd`) — fixed template, best plddt |

**⚠️ Correction (2026-07-27):** an earlier version of this table listed a "Folded
(predicted) **antigen only** → 2/15" rung. That is a **mislabel** — verified against the
input JSONs and score files. **No predicted-antigen-only run was ever done.** The two 2/15
results (`expb`, `expc`) both fed *both* chains as predicted templates (expb template JSON =
3/3 chains templated; expc = 2/2). The 0/15 result (`expd`) *also* fed both chains. So the
2-vs-0 gap among predicted runs is **not** antigen-only-vs-both — it's the **fold source**
(isolation / diverse folds → 2; single fixed best-pLDDT pose → 0). The only run that
templated the antigen *alone* (antibody free) is `agonly`, and it used the **native**
antigen → 8/15. Rescue counts re-tallied from the score JSONs under selection
(top-interface-pair_iptm pose's DockQ ≥ 0.23) confirm: agonly 8, expb 2, expc 2, expd 0.

### How each predicted-fold template was actually built (verified from builders + coords)

The three predicted-fold runs differ on two independent axes — **fold source**, **fixed vs
varying across the 50 generation seeds** — and on **whether the template was quality-selected**.
All three feed *both* chains (Ab + Ag) a template.

| Run | Fold source | Candidate pool per chain | Selection | Across 50 gen seeds | Rescued |
|---|---|---|---|---|---|
| **Exp B** (`expb`) | isolation folds (`mono_out`) | 4 seeds (501–504) × 3 samples = **12** | **best `chain_pLDDT`** (single fold) | **fixed** (same template all 50) | 2/15 |
| **Exp C** (`expc`) | free-run complex poses | the 5 diffusion samples of each seed | **none** — takes **`sample_0`** | **varies** (gen-seed *i* ← free-run seed *i*) | 2/15 |
| **Exp D** (`expd`) | free-run complex poses | all 50 seeds × 5 samples | **best global `chain_pLDDT`** per chain (mix-and-match Ab/Ag across seeds) | **fixed** (same template all 50) | 0/15 |

- **Exp B — fixed, quality-selected, isolation.** Verified by exact CA-coordinate match: 7z2m's
  antibody-q0 template (`mmcif/y722.cif`) is `mono_out/7z2m_mono_q0/seed_504/sample_1`. Note a
  separate 50-seed isolation run exists (`mono50_out`, 25–50 seeds/chain) but was **never wired
  into a rescue run** — Exp B's template matched `mono_out` (4-seed), none of `mono50_out`.
- **Exp C — varying, NOT quality-selected.** `build_expc.py` uses
  `{pdb}/seed_{s}/predictions/{pdb}_sample_0.cif` — the *first* diffusion sample of each seed, an
  arbitrary draw, **not** the best of the 5. So C's per-seed template quality is essentially
  random.
- **Exp D — fixed, quality-selected, complex.** `build_expd.py` picks the highest global
  `chain_pLDDT` pose per chain across all free-run poses, mix-and-match (best Ab pose + best Ag
  pose, possibly different seeds), one fixed template for all 50 gen seeds.

**C vs D (same fold source = free-run poses):** varying (C) 2/15 > fixed (D) 0/15 — and D loses
*despite* using the nominally higher-quality (best-pLDDT) template. But it's a **weak** signal
(2 vs 0, both ~failures) and **confounded**: C and D also differ in selection (C = unselected
`sample_0`, D = best-pLDDT). "Fixed" alone isn't the killer either — Exp B is also fixed and gets
2/15; the fixed-isolation vs fixed-complex gap (2 vs 0) shows fold source matters at least as much.

**Untested clean arm:** *diverse **and** quality-selected* — for each gen seed, template with the
**best-`pLDDT` of that seed's 5 samples** (C's diversity without C's random draw). Never run.

**Interpretation:** the confidence head is hypersensitive to sub-Å template accuracy. Only
an essentially *exact* fold collapses the false ipTM. Predicted folds — even good ones
(~0.94 global lDDT) — are wrong precisely in CDR-H3 (antibody) / the epitope surface
(antigen), so they carry ~zero corrective signal. Feeding predicted folds for *both* chains
never beats the native-antigen-only 8/15 (best predicted result = 2/15).

**Open (never run):** the true apples-to-apples partner of native-antigen-only's 8/15 —
**predicted antigen only, antibody free, 50 seeds** (reuse the `agonly` harness, swap the
native antigen template for the `mono_out` predicted fold). This is the clean "8/15 → ?"
antigen-side number; the "8 → 2 antigen-side echo" story was based on the mislabeled row and
is **not yet supported** by a matched experiment.

Exp D even pulls a few cases *below* the free-run pool best (imperfect template nudges
generation off the good basin: 8fas 0.99 → 0.20, 8che_AE 0.52 → 0.11).

## 5. Dead avenues (ruled out this project)

- **Isolation folding** (chain / Fab H+L folded without antigen) gives **no** fold-accuracy
  gain over free complex runs (antibody dead-even; antigen *worse*, esp. peptides that need
  the antibody scaffold). The docking-distortion hypothesis is refuted.
- **Self-template** (using the model's own prediction as its template) is **net-harmful** —
  breaks good seeds more than it rescues decoys.
- **Cross-model agreement** (ESM vs Protenix) — dead; errors are correlated.
- **H3-pLDDT vs global-pLDDT pose selection** — 9/12 return distinct poses but they're
  near-ties; only 8q3j is a clear win (0.61 → 0.90). Not a general fix.
- **Aggregation** (median/medoid/weighted) — loses top-1 via regression-to-median.
  Best selection method remains "leaders" (greedy structural NMS on ipTM, no aggregation).

## 6. Through-line

Fold quality and generation were **never** the bottleneck — free runs already have both
near-native chain folds *and* 15/15 pose coverage. The wall is **(a) selection** + **(b) the
exact-fold hypersensitivity of the confidence head**, whose failure is driven by the one
region deployable folds get wrong: **long CDR-H3 loops**. Long-H3 confident-decoy interfaces
are simultaneously the hardest to select and the hardest to generate correctly.

---

## Open threads

- **Predicted-antigen-only** (antibody free, predicted antigen template, 50 seeds) — NOT YET
  RUN. The clean partner to native-antigen-only (8/15). Reuse `agonly` harness, swap native
  antigen template → `mono_out` predicted fold. Gives the real "8/15 → ?" antigen-side number
  (the previously-claimed 2/15 for this was a mislabel of expb's both-chain result).
- **Exp D2** (both chains from single best-overall pose) — generating; expected ~0/15,
  confirms per-chain mix-and-match wasn't the culprit.
- **CDR-H3 coordinate graft** — graft the highest-per-residue-pLDDT H3 onto the best
  framework, then use as template. Untested; the one lever that targets the actual error.
- **Oracle-best H3 within pool**: is the pool's best-H3 pose good (→ selection, graftable) or
  is even it wrong for long loops (→ generation-limited)? Quick follow-up on existing data.
- **Distogram head / paratope-plausibility** re-rankers — untested top-1 levers.

## Key paths

- Session scratch: `.../scratchpad/selftmpl/` — builders (`build_expd.py`, `build_expd2.py`),
  scorers (`score_expd.py`), fold comparison (`fold_cmp2.py`), `mech.py`, `meta.json`.
- Canonical OST scorer (always match its flags): `/home/ebrahim/tts_abag/eval/eval_ost_foldbench.py`.
- Natives: `/home/ebrahim/FoldBench/examples/ground_truths/`.
- Free predictions: `/home/ebrahim/Protenix/foldbench_abag/predictions_template/`.
- Relevant memories: `native-rescue-mechanism`, `native-template-flips-decoy`,
  `error-localizes-to-cdrh3`, `isolation-folds-not-better`, `self-template-fails`,
  `confidence-adaptive-selection`, `top1-selection-frontier`.

---

## Session addendum (2026-07-28) — deployable-selection sweep

Full results + numbers: `confident_decoy_experiments/RESULTS.md`. Headlines:

- **abonly 5/15 (goods 15/15 SAFE), agonly 8/15 — DISJOINT sets, 0 overlap.** Each decoy is
  bottlenecked on ONE chain: antigen×8, antibody×5, both×1 (8the), unrescuable-by-native×1 (8fas).
- **oracle (best-lDDT fold per chain) = 8/15**, brackets Exp D (0) and native (14):
  antigen side is **selection**-limited (good fold exists, pLDDT blind, corr≈+0.01), antibody side is
  **generation**-limited (best paratope fold only ~0.86 lDDT, oracle gets just 1/5 antibody cases).
- **pLDDT never picks the oracle fold (0/34 chains)** — harmless for antibodies (tight pool 0.055),
  fatal for antigens (wide pool 0.121, good fold is a rare outlier; consensus corr +0.19, still fails).
- **Mechanism: ipTM is gated by interface FOLD QUALITY.** Good fold RAISES the good pose's ipTM;
  only EXACT fold also CRASHES the decoy (wrong dock becomes geometrically impossible). Deployable
  folds do the "raise good" half only (oracle raised good ipTM 5/9, crashed decoy 1/9).
- **"Which single pose's fold" ALL fail — mix-and-match oracle is the ceiling:** topiptm (decoy pose's
  fold) 0/5 *reinforces* the decoy (good-pose ipTM unchanged); topdockq (best-docked pose's fold)
  1/14 (best-DOCKED ≠ best-FOLD; fold/dock decoupled); sposefv (same-pose Fv) ≈ oracle, not better.
- **Through-line: confident decoys "all the way down"** — generation gives both the right dock AND
  near-native folds; the confidence heads (ipTM for pose, pLDDT for fold) can't select them.
  **Only open deployable lever: a better-than-pLDDT ANTIGEN-fold selector (rare-outlier problem).**
- Dead ends this session: same-pose-Fv, decoy-pose-fold, best-dock-pose-fold, pLDDT/consensus
  fold-selection. Bug: oracle_goods (chain-name PDB-write) — abonly_goods 15/15 already covers safety.
