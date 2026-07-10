# Selecting the good dock on confident-decoy traps by enumerating hypotheses

_First real finding, 2026-07-10._ A label-free way to recover the correct antibody–antigen pose on
"confident-decoy" interfaces — where ipTM confidently selects the wrong structure — by treating the
seed pool as a small set of competing structural hypotheses instead of scoring poses individually.

## The problem

On a confident-decoy interface a good dock (DockQ > 0.4) exists in the seed pool, but the
single highest-ipTM pose is a decoy (DockQ < 0.23). Naive ipTM selection (`argmax ipTM`) therefore
fails by construction. Prior project work established this is a **genuine ipTM failure, not
undersampling** (depth to 1024 seeds does not rescue it; see
[`esmfold2fast_tts_benchmark.md`](esmfold2fast_tts_benchmark.md)).

## Data

Seven tight-margin ab–ag interfaces (`T7`): 8ts0 A-C, 8ts0 B-C, 8a44 A-D, 7yxu A-C, 8a44 A-C,
7zoz A-C, 8vui B-C. For each, `matrix_iptm.py` builds a **1025×1025 seed×seed DockQ matrix**
(Option A: one representative pose per seed = its best-ipTM diffusion sample; the mask0/dropout0
`baseline_seed` is included as a row; DockQ is interface-restricted to c1–c2). Matrices live at
`results/iptm_matrix/matrix_<cx>_<c1><c2>.npy` with a row-index CSV (seed, sample, ipTM, dockq_gt).
`dockq_gt` is used **only to score** whether a pick is good — never to select.

Reproduce all numbers below: `python steer/experiments/hypothesis_enumeration.py 0.5`

## What did NOT work (for the record)

Every method that rewards **confidence or structural prominence** picks the decoy, because on these
traps the decoy is *both* more confident and the denser/larger structural basin:

| method | good docks recovered / 7 |
|---|:--:|
| naive ipTM (`argmax`) | 0 |
| DockQ-weighted consensus ipTM — sum `S` or average `A` (soft, no cutoff) | 1 |
| thresholded consensus / greedy-cluster mass×mean (symmetric, unsupervised) | 1 |

Measured cause on 8ts0 A-C: among top-50 ipTM poses the decoys are mutually similar (avg pairwise
DockQ 0.56, ~559 neighbours each) while the good docks are diffuse (0.48, ~90 neighbours). The
confident decoy is a **sharp reproducible attractor**; consensus/mass amplify it. The earlier
"oracle 4/7" (mass×mean) leaned on two crutches that do not transfer: the all-samples pool, and an
**asymmetric measurement** (good cluster counted generously by ground-truth membership, decoy
counted stringently near one reference).

## What worked — Method 1: first distinct hypothesis (7/7)

Descend seeds by ipTM. `H1` = most confident seed (the decoy). Keep going and take `H2` = the first
seed that is **genuinely structurally distinct** from H1 (`DockQ(H1, seed) < τ`). On a
confident-decoy trap the model proposes essentially two confident hypotheses — the decoy and the
good dock — and **H2 is the good dock**.

Rule: descend by ipTM, skip near-duplicates of the top pick, return the first different structure.

| interface | H2 ipTM-rank | DockQ(H2,H1) | DockQ_gt of H2 |
|---|:--:|:--:|:--:|
| 8ts0 A-C | 20 | 0.027 | 0.908 ✅ |
| 8ts0 B-C | 28 | 0.029 | 0.866 ✅ |
| 8a44 A-D | 9 | 0.027 | 0.640 ✅ |
| 7yxu A-C | 2 | 0.007 | 0.826 ✅ |
| 8a44 A-C | 22 | 0.033 | 0.698 ✅ |
| 7zoz A-C | 2 | 0.088 | 0.605 ✅ |
| 8vui B-C | 100 | 0.054 | 0.720 ✅ |

**7/7**, and **robust to the threshold**: identical 7/7 for every τ in [0.10, 0.40] (any cutoff that
means "a different interface"). It works because it never tries to out-score the decoy on
confidence — it uses ipTM only to *rank* and DockQ only to test *distinctness*.

## What partly worked — Method 2: cluster-confidence (3/7 significant)

Anchor two clusters on H1 and H2 (each seed joins the nearer reference if within a DockQ floor τ),
then ask: is the **good** hypothesis's cluster more confident than the decoy's? (Your
variance-reduction idea: pool ipTM over the cluster to get a lower-variance confidence estimate.)

At τ=0.5, by cluster **median** ipTM (clusters are ~100% pure):

| interface | winner | Cliff's δ | MWU p | significant? |
|---|:--:|:--:|:--:|:--:|
| 8a44 A-D | H2 (good) | −0.88 | 2e-4 | **yes** |
| 7yxu A-C | H2 (good) | −0.86 | 2e-9 | **yes** |
| 8a44 A-C | H2 (good) | −0.76 | 2e-3 | **yes** |
| 8vui B-C | H2 (good) | −0.04 | 0.75 | no (tie) |
| 7zoz A-C | H1 (decoy) | +0.33 | 0.56 | no (n=2/3) |
| 8ts0 A-C | H1 (decoy) | +0.19 | 4e-3 | yes |
| 8ts0 B-C | H1 (decoy) | +0.34 | 2e-10 | yes |

**H2 wins the median comparison on 4/7; 3/7 are statistically significant** (8vui's win is a
coin-flip, δ=−0.04). Two mechanistic points:

- **Pooling vindicated where it works.** On 8a44 A-C the single decoy pose (ipTM 0.582) looks more
  confident than the good pose (0.553), but the cluster medians flip (decoy 0.434 vs good 0.540):
  the confident decoy was a **lucky high outlier in a mediocre cluster**, while the good hypothesis
  is *consistently* confident. Variance reduction exposes this. **Median beats mean** (discounts the
  good cluster's low tail); **tight τ (≥0.4) matters** (loose τ contaminates the good cluster).
- **Mass hurts here** (unlike the oracle). H2 wins mass×median only 2/7. Reason: the honest,
  unsupervised H2 cluster is usually *smaller* than the decoy cluster, so mass favours the decoy.
  The earlier "4–5/7 with mass" relied on the ground-truth cheat inflating the good cluster's size.
  Median-only (3/7 significant) is the better honest rule. Interesting complementarity: median wins
  the *small* good-basin cases (8a44, 6 good seeds), mass×median wins the *large* good-basin cases
  (8ts0, ~230 good seeds) — union covers 6/7, but no single fixed rule does.

## Synthesis

- The confident-decoy trap is a **two-hypothesis** problem. Don't score poses — enumerate the
  distinct confident hypotheses. The good dock is the top confident structure that *disagrees* with
  the winner (**H2**).
- **Best selector: first-distinct → return H2. 7/7, threshold-robust.**
- Confidence *pooling* is a real but partial *confirmation* signal (3/7 significant), useful when the
  good hypothesis is robustly confident; it cannot help on **8ts0**, the one genuinely hard case
  where the decoy is both structurally dominant and truly more confident.

## Caveats (do not overclaim)

1. **These 7 are the analysis set, not held out.** The 7/7 was found by studying these interfaces
   (Method 1 was literally derived from 8ts0's plot). Overfitting risk is real and unquantified.
2. **Method 1 is trap-specific.** On all 7, H1 is a decoy. On a normal interface where ipTM already
   works, "return H2" would return a decoy. The general deliverable is "the good dock is H1 or H2"
   (return both); H2 is the answer only when you know/ suspect a trap. A trap detector is future work.
3. Pool caveat: Option A (best-ipTM sample per seed) can suppress good docks that lose the
   intra-seed ipTM competition; the all-samples pool (Option B) is untested here and may differ.

## Next: the decisive experiment

Validate blind on **held-out confident-decoy interfaces** from `decoy_structural_analysis.csv` that
we have not looked at: build their DockQ matrices (`matrix_iptm.py`), run
`hypothesis_enumeration.py` with no peeking. If first-distinct holds near 7/7 on fresh interfaces,
it is a real method for this trap class; if it collapses, it was overfit to these 7.

## Files

- `steer/experiments/matrix_iptm.py` — builds the seed×seed DockQ matrices (memory-safe).
- `steer/experiments/hypothesis_enumeration.py` — both methods + Mann-Whitney; reproduces all numbers.
- `results/iptm_matrix/matrix_*.npy` (+ `_index.csv`, `scores_*.csv`) — the matrices and per-seed scores.
