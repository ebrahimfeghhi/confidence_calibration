# Selecting the good dock on confident-decoy traps by enumerating hypotheses

_First real finding, 2026-07-10._

A label-free way to recover the correct antibody–antigen pose on "confident-decoy" interfaces —
where ipTM confidently selects the wrong structure — by treating the seed pool as a small set of
competing **structural hypotheses** instead of scoring poses individually.

## The problem

On a confident-decoy interface a good dock (DockQ > 0.4) exists in the seed pool, but the single
highest-ipTM pose is a decoy. Naive ipTM selection (`argmax ipTM`) therefore fails **by
construction** on this set. Prior work established this is a genuine ipTM failure, not
undersampling — depth to 1024 seeds does not rescue it (see
[`esmfold2fast_tts_benchmark.md`](esmfold2fast_tts_benchmark.md)).

## The method — one method, two stages

Do not think of these as rival methods. **Stage 1 decides what the candidate answers are; stage 2
decides which one to believe.** Both are needed, and they are measured by different metrics.

**τ = 0.4 is the single parameter.** It is used for both stages.

### Stage 1 — ENUMERATE (greedy non-maximum suppression)

Descend seeds by ipTM. A seed becomes a new hypothesis `H_k` only if it is structurally
**distinct** from *every* hypothesis already accepted: `DockQ(H_i, H_k) < τ` for all `i < k`.
(H3 must be < τ to both H1 and H2; H4 to all three; and so on.) This compresses ~1025 seeds into a
handful of competing structural hypotheses.

- Metric: **recall@H** — is a good dock among the top-H hypotheses?
- No minimum-cluster / support rule. Good docks can be genuinely rare — 8a44's good hypothesis has
  only **6 seeds of 1025**. Any support floor would delete the correct answer.

### Stage 2 — SCORE (pooled confidence)

Assign each seed to its **nearest anchor** (the hypothesis it has the highest DockQ to, provided
that DockQ ≥ τ; seeds close to nothing stay unassigned). Then score each hypothesis by **pooling
ipTM over its cluster** (median). Pooling is a variance-reduced confidence estimate: it asks *"is
this hypothesis consistently confident?"* rather than *"did one pose get a lucky high ipTM?"*

- Metric: **precision@1** — is the top-*scored* hypothesis a good dock?

Reproduce everything below: `python steer/experiments/hypothesis_enumeration.py 0.4`

## Results (τ = 0.4, all 7 interfaces)

### Stage 1: enumeration → recall@2 = 7/7

| interface | ipTM rank of each enumerated hypothesis | hypothesis-depth of first good dock |
|---|---|:--:|
| 8ts0 A-C | 1, **20**, 54, 108, 986 | 2 |
| 8ts0 B-C | 1, **28**, 121, 986, 996 | 2 |
| 8a44 A-D | 1, **9**, 216, 224, 231 | 2 |
| 7yxu A-C | 1, **2**, 5, 49, 53 | 2 |
| 8a44 A-C | 1, **22**, 213, 223, 226 | 2 |
| 7zoz A-C | 1, **2**, 3, 4, 5 | 2 |
| 8vui B-C | 1, **100**, 214, 260, 280 | 2 |

**recall@1 = 0/7** (by construction — H1 is the argmax-ipTM pose, which is a decoy by definition of
this set). **recall@2 = 7/7. recall@3 = 7/7.** On every interface the good dock is the *second*
distinct hypothesis: the model proposes essentially two confident structural stories, and the good
dock is the top confident structure that **disagrees** with the winner.

Note how wide a net this casts cheaply: on 8vui the good dock sits at raw ipTM **rank 100**, and on
8ts0 at rank 20 — ranks no confidence-based shortlist would reach — yet both are **hypothesis #2**,
because everything above them is a near-duplicate of the decoy.

### Stage 2: pooled confidence → precision@1 = 5/7

Which hypothesis do we return if we must pick exactly one?

| pooling rule for hypothesis confidence | good docks picked / 7 |
|---|:--:|
| `top` — max ipTM in the cluster (≈ no pooling; the naive baseline) | **0/7** |
| `mean` — mean ipTM over the cluster | 3/7 |
| **`median` — median ipTM over the cluster** | **5/7** |
| `mass × median` — cluster size × median | 1/7 |

The `top` row is 0/7 by construction (it reduces to argmax ipTM). The real result is that
**pooling confidence over the hypothesis lifts precision@1 from 0/7 to 5/7.** Median beats mean
(it discounts the good cluster's low tail). **Mass actively hurts** (1/7): the honest, unsupervised
good hypothesis is usually *smaller* than the decoy's, so weighting by size favours the decoy. An
earlier "4–5/7 with mass" result relied on ground truth inflating the good cluster's size.

Is the good hypothesis' pooled confidence *significantly* higher than the decoy's?

| interface | good hyp size | good median ipTM | decoy hyp size | decoy median ipTM | Cliff's δ | MWU p | significant? |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 8a44 A-D | 7 | 0.489 | 327 | 0.345 | +0.62 | 5e-3 | **yes** |
| 7yxu A-C | 45 | 0.268 | 27 | 0.130 | +0.81 | 1e-8 | **yes** |
| 8a44 A-C | 6 | 0.540 | 290 | 0.406 | +0.78 | 1e-3 | **yes** |
| 8vui B-C | 26 | 0.446 | 264 | 0.442 | −0.06 | 0.59 | no (tie) |
| 7zoz A-C | 17 | 0.132 | 17 | 0.137 | −0.22 | 0.29 | no |
| 8ts0 A-C | 159 | 0.512 | 19 | 0.515 | −0.21 | 0.13 | no |
| 8ts0 B-C | 233 | 0.515 | 1 | 0.537 | −1.00 | 0.09 | no |

**3/7 significant.** The mechanism is visible on 8a44 A-C: the single decoy pose (ipTM 0.582) looks
*more* confident than the single good pose (0.553), but the cluster medians **flip** (decoy 0.406 vs
good 0.540). The confident decoy was a **lucky high outlier in a mediocre cluster**, while the good
hypothesis is *consistently* confident. Variance reduction is what exposes this.

**8ts0 (both interfaces) is the genuinely hard case** and the honest failure: there the decoy is not
a lucky outlier — it is structurally dominant *and* truly more confident, pose-wise and pooled.
Pooling cannot fix it. Stage 1 still recovers it (it is hypothesis #2), which is exactly why the
shortlist matters.

## τ — the single parameter

τ is a DockQ **similarity** threshold (high DockQ = structurally similar). The same τ appears in both
stages, but the two stages compare against it in **opposite directions** — so they want opposite
*values*. This is the one subtlety worth internalising.

**Stage 1, enumeration — the test is `DockQ(H_i, H_k) < τ`, so τ is an UPPER bound.** A candidate
must be *below* τ in similarity to every accepted hypothesis to count as new.
- **Small τ ⇒ strict.** Only radically different structures qualify as new hypotheses. Safe.
- **Large τ ⇒ permissive.** A pose with DockQ 0.5 to H1 — plainly the same basin, just shifted —
  passes a `< 0.6` test and gets promoted to "new hypothesis". Those spurious hypotheses occupy
  shortlist slots and push the real good dock deeper. This is why recall@2 falls from 7/7 to 5/7
  when τ goes 0.4 → 0.5.

**Stage 2, assignment — the test is `DockQ(anchor, seed) ≥ τ`, so τ is a LOWER bound.** A seed must
be *above* τ in similarity to join a cluster.
- **Large τ ⇒ strict.** Only genuinely similar seeds join. Clusters stay pure, pooled ipTM is clean.
- **Small τ ⇒ permissive.** Almost every seed clears a `≥ 0.1` bar and gets swept into some cluster,
  including seeds that barely resemble the anchor. The cluster bloats, and its pooled ipTM regresses
  toward the mean of the whole pool — erasing the very signal we are pooling to find.

Concretely, on 8a44 A-C the true good basin is 6 seeds. At τ=0.4 the good cluster is exactly those
**6 seeds, median ipTM 0.540** — and it beats the decoy (0.406). At τ=0.23 the same cluster bloats to
**13 seeds and its median collapses to 0.154**: seven contaminating seeds dragged the good
hypothesis' confidence down by a factor of 3.5 and it loses. Same τ, same data — the low value just
destroyed stage 2.

So: **small τ is safe for enumeration but poisonous for assignment; large τ is safe for assignment
but poisonous for enumeration.** They are fighting over one number.

| τ | recall@1 | recall@2 | recall@3 | precision@1 (median pooling) |
|:--:|:--:|:--:|:--:|:--:|
| 0.10 | 0/7 | 7/7 | 7/7 | 1/7 |
| 0.15 | 0/7 | 7/7 | 7/7 | 2/7 |
| 0.23 | 0/7 | 7/7 | 7/7 | 2/7 |
| 0.30 | 0/7 | 7/7 | 7/7 | 3/7 |
| **0.40** | 0/7 | **7/7** | **7/7** | **5/7** |
| 0.50 | 0/7 | 5/7 | 6/7 | 5/7 |
| 0.60 | 0/7 | 5/7 | 5/7 | 4/7 |

**τ = 0.4 is the corner**: the last value where recall@2 is still 7/7, and the first where precision
peaks. Recall degrades above it (0.5 → 5/7, as barely-shifted decoys start counting as new
hypotheses); precision degrades below it (0.23 → 2/7, as clusters get contaminated). A single τ
does satisfy both stages here, but only just — the two stages want opposite things, and if they
ever stop overlapping on held-out data, decoupling into τ_enumerate and τ_assign is the fix.

## Synthesis

- The confident-decoy trap is a **two-hypothesis** problem. Don't score poses — enumerate the
  distinct confident hypotheses, then score *those*.
- **Enumeration is the strong part: recall@2 = 7/7.** Compressing ~1025 seeds to a 2-item shortlist
  that always contains the answer is the robust, deployable result.
- **Pooled confidence is the useful-but-partial part: precision@1 = 0/7 → 5/7**, 3/7 significant.
  It works when the decoy is a lucky outlier; it fails when the decoy is genuinely, consistently
  confident (8ts0).
- If you must return one structure, return the top-scoring hypothesis (5/7). If you can return two,
  return the shortlist (7/7).

## Caveats (do not overclaim)

1. **These 7 are the analysis set, not held out.** The method was *derived* by studying these
   interfaces. Overfitting risk is real and unquantified. τ=0.4 in particular is now selected on
   this set — it must be locked before validation, not re-tuned on held-out data.
2. **The method presumes a trap.** recall@1 = 0/7 and the `top` rule's 0/7 are both *by
   construction*: these interfaces were selected as ones where argmax-ipTM fails. On a normal
   interface where ipTM already works, H1 is the right answer and pooling could talk you out of it.
   A **trap detector** — deciding whether you are in this regime at all — is unbuilt and is the
   main gap between this and a general selector.
3. Pool caveat: Option A (best-ipTM sample per seed) can suppress good docks that lose the
   intra-seed ipTM competition. The all-samples pool (Option B) is untested.

## Next: the decisive experiment

Lock **τ = 0.4** and validate blind on **held-out confident-decoy interfaces** from
`decoy_structural_analysis.csv` that we have not looked at: build their DockQ matrices
(`matrix_iptm.py`), run `hypothesis_enumeration.py`, no peeking, no re-tuning. Report **recall@2,
recall@3, and precision@1**. If recall@2 holds near 7/7 on fresh interfaces it is a real method for
this trap class; if it collapses it was overfit to these 7.

## Data

Seven tight-margin ab–ag interfaces (`T7`): 8ts0 A-C, 8ts0 B-C, 8a44 A-D, 7yxu A-C, 8a44 A-C,
7zoz A-C, 8vui B-C. For each, `matrix_iptm.py` builds a **1025×1025 seed×seed DockQ matrix**
(Option A: one representative pose per seed = its best-ipTM diffusion sample; the mask0/dropout0
`baseline_seed` is included as a row; DockQ is interface-restricted to c1–c2). Matrices live at
`results/iptm_matrix/matrix_<cx>_<c1><c2>.npy` with a row-index CSV (seed, sample, ipTM, dockq_gt).
`dockq_gt` is used **only to score** whether a pick is good — **never to select**.

## Files

- `steer/experiments/matrix_iptm.py` — builds the seed×seed DockQ matrices (memory-safe).
- `steer/experiments/hypothesis_enumeration.py` — the method (both stages) + stats; reproduces all
  numbers above.
- `results/iptm_matrix/matrix_*.npy` (+ `_index.csv`, `scores_*.csv`) — matrices and per-seed scores.
