# Steering ESMFold2-Fast out of wrong antibody-antigen docks — project summary

Written 2026-07-01 as a snapshot for external review. Self-contained: no assumed
familiarity with the codebase.

## 1. The problem

`ESMFold2-Fast` (an AlphaFold3-style model: an ESMC language model feeds a
recurrent "pairformer" trunk, which recycles ~17 times before a diffusion head
decodes 3D coordinates) sometimes folds an antibody-antigen (ab-ag) complex with
the antibody docked in the wrong place or orientation on the antigen. For a
sizeable set of real ab-ag targets, the *default* fold is confidently wrong
(high model-reported confidence, low true DockQ against the known structure),
and resampling with different random seeds essentially never escapes it — DockQ
stays below 0.23 (an approximate "wrong dock" cutoff) no matter how many plain
seeds you try.

Goal: find a way to steer the model into the *correct* dock, ideally:
- **Label-free** — no access to the true structure at steering time (only usable
  for scoring/validation after the fact).
- **Generalizable** — no per-target hand-tuning (no hardcoding "mask these
  residues" or "target this specific structure"), so it works on a new ab-ag
  pair without bespoke setup.

## 2. The method that works: label-free sequential greedy self-repulsion

Core idea: run the model repeatedly from slightly different starting points,
and after each attempt, explicitly steer the *next* attempt's internal state
away (in distogram space) from every attempt made so far. No ground truth is
used in the loss — only the model's own prior outputs. Over enough attempts,
this pushes the model out of its default wrong-dock basin and into others,
including (sometimes) the correct one.

Concretely, per attempt ("run"):
1. Start from the model's baseline LM single-representation `s_lm` (shape
   `[1, L, 256]`, one 256-dim vector per residue), nudged by a small random
   perturbation.
2. Run a short forward pass (K=3 recycling loops, not the full 17) to get a
   **cross-distogram**: softmax'd, binned pairwise-distance predictions
   restricted to the antibody-rows × antigen-columns block of the full
   L×L distogram (i.e., only inter-chain / interface distances).
3. Gradient-**ascend** `s_lm` for M=5 steps to *maximize* the Wasserstein-1
   (W1) distance between this attempt's cross-distogram and the mean of every
   prior attempt's cross-distogram ("anchors"). This is the only place any
   optimization happens; everything else is a fixed forward pass.
4. Append this attempt's final cross-distogram to the anchor list. Repeat.
5. Periodically (every 16 attempts), run the *optimized* `s_lm` through the
   full 17-loop recycling + diffusion decode to get real 3D structures, score
   them against the true structure (DockQ) — this is for monitoring/stopping
   only, not part of the loss.

Proven hyperparameters: `M=5` steps, `lr=0.3`, `K=3`, initial nudge
`init_frac=0.05` (fraction of baseline `s_lm`'s norm), loss = mean W1 distance
to all anchors. This was validated on an original test structure (`7txw`) where
it reliably escapes the wrong basin (DockQ ≈ 0.85-0.92, matching a fully
supervised/ground-truth-guided version of the same idea).

## 3. The 6-target campaign (2026-06-30 → 07-01)

Ran this recipe as a long-lived "campaign" process (anchors accumulate for up
to 12h in one process; batch-decode+score every 16 attempts; stop early on a
DockQ ≥ 0.23 hit or the 12h budget) across **6 real ab-ag targets** that all
fail with plain resampling (best baseline DockQ < 0.23 for all six). Two
targets per GPU, sequential, across 3 GPUs.

**Result: 3 of 6 rescued.**

| target | outcome | best DockQ | notes |
|---|---|---|---|
| `8jyr` | **HIT** | 0.952 | clean — all 5 decoded samples from the winning attempt scored 0.90-0.95 |
| `8urf` | **HIT** | 0.378 | weak — only 1 of 5 samples from the winning attempt hit; ipTM 0.61 |
| `8vzo` | **HIT** | 0.544 | weak — only 1 of 5 samples hit; ipTM only 0.37 |
| `8im0` | no hit | 0.047 | ran the full 512-attempt cap; a previously-known "uncrackable" control target |
| `8vvl` | no hit | 0.167 | hit the 12h time budget; closest miss of the non-hits |
| `8z8v` | no hit | 0.070 | hit the 12h time budget; no real traction |

### 3a. A confidence-selection caveat

In a real pipeline you'd pick the best decoded sample by model confidence
(ipTM), not by DockQ (which needs ground truth). Checking this after the fact:

- For `8jyr`, ipTM works perfectly — the 5 highest-ipTM samples out of 160 total
  decoded across the whole campaign are exactly the 5 samples of the winning
  attempt.
- For `8urf`, ipTM **fails badly** — the winning sample ranks 351st out of 400
  by ipTM. Several *wrong* decoys score ipTM 0.86-0.87, well above the true
  hit's 0.61.

So this method can find a real rescue that the model itself doesn't recognize
as good. That's a real gap between "we found it in hindsight (DockQ)" and "a
deployed system would have surfaced it (ipTM)." An earlier, larger-scale check
on `7txw` (n=2060 samples) had found ipTM-based selection to work very well
(AUC 0.945) — so this failure mode doesn't generalize from that earlier result
and seems specific to these weaker, single-sample hits.

### 3b. Root-causing the 3 non-hits: orientation problem, or misfolding?

The steering loss only ever touches the antibody×antigen (interface) block of
the distogram — never the antibody's own fold or the antigen's own fold. So it
can only fix *wrong orientation*, never *wrong intra-chain structure*. Checked
whether any of the 3 non-hits actually have a misfolded antigen (which this
method could never fix), using antigen-only structural agreement (lDDT,
ignoring the antibody chain and the interface entirely) against the true
antigen structure:

| target | antigen-only lDDT vs. true structure |
|---|---|
| `8jyr` (hit) | 0.924 |
| `8z8v` (no hit) | 0.907 |
| `8im0` (no hit) | 0.863 |
| `8urf` (hit) | 0.859 |
| `8vzo` (hit) | 0.792 |
| **`8vvl` (no hit)** | **0.208** |

Only `8vvl` shows real misfolding — its antigen barely resembles the true fold
at all, while `8im0` and `8z8v`'s antigens fold about as well as the
successfully-rescued targets'. So of the 3 failures, 2 (`8im0`, `8z8v`) are
plausibly still "just" hard orientation problems the method could in
principle solve with more search; only `8vvl` needs a fundamentally different
fix.

Bonus finding: this doesn't require ground truth to detect. The model's own
per-residue confidence (pLDDT, already embedded as the B-factor column in
every decoded structure file) shows the same signal — `8vvl`'s antigen chain
averages pLDDT ≈ 30 (its own antibody chains: ≈ 87-91) vs. `8jyr`'s antigen at
≈ 80-84. A ~50-60 point pLDDT gap between an antigen and its own antibody
chains, using zero external information, looks like a usable red flag for "this
target needs more than interface steering" before spending a 12h campaign on
it.

## 4. Extensions currently running (as of 2026-07-01 evening)

### 4a. Jointly optimizing z0 alongside s_lm

The recurrent pairformer trunk needs an initial pair-state `z0` (shape
`[1, L, L, 256]` — one 256-dim vector *per residue pair*, not per residue) to
seed its 17 recycling loops; today it's drawn once from a fixed truncated-normal
distribution and never touched by the optimizer. Since `z0` already sits on the
exact same forward/backward computation graph as `s_lm` (it's just consumed
differently — `s_lm` re-enters every loop, `z0` only seeds the first one),
optimizing it jointly costs no extra forward or backward passes — only a
second Adam parameter group. The real cost is memory: `z0` is O(L²) versus
`s_lm`'s O(L), so its optimizer state is much bigger (~1.6-4GB depending on
target size, vs. essentially nothing for `s_lm`).

First attempt used the same learning rate (0.3) for both — `z0` immediately
overshot to ~11x its own norm within 3 steps (effectively erasing the original
initialization), while `s_lm` stayed in its normal, stable range. Giving `z0`
its own 30x-smaller learning rate (0.01) fixed this — its relative movement per
step now roughly matches `s_lm`'s.

Currently running: the 3 successful targets (`8jyr`, `8urf`, `8vzo`)
sequentially on one GPU with joint `z0`+`s_lm` optimization, same loss and
budget as the original campaign otherwise. Open question: does this turn the 2
weak, single-sample hits into cleaner, more-robust ones (or find them faster)?

### 4b. Adding an intra-antigen loss term, targeted at `8vvl`

Since `8vvl` was diagnosed as a genuine intra-chain misfolding problem (not
orientation), extended the loss to optionally also include the antigen×antigen
block of the distogram (previously never used), alongside the proven
antibody×antigen block. This is also compute-free in the same way as 4a — the
model already computes the *entire* L×L distogram every forward pass; using
more of it in the loss doesn't cost extra forward/backward compute, only makes
the stored anchor tensors bigger.

Currently running two sequential 12h attempts on `8vvl` with this extended
loss: one with joint `z0`+`s_lm` (4a combined with this), one `s_lm`-only — to
see whether making the loss "aware" of the antigen's own fold quality can
rescue this specific target.

### 4c. Baseline control sweep

Running 1024 unsteered baseline seeds (no repulsion at all, just the standard
10%-mask/30%-dropout sampling recipe) each for `8jyr` and `8urf` — the two
targets where steering found a hit — to check whether either hit could plausibly
have been found by scale/luck alone, without any steering. This is the
sanity-check control for the method's core value proposition.

## 5. Open questions / where feedback would help

1. **Per-step loss shape.** On the original `7txw` recipe validation, the
   repulsion loss visibly plateaus within the 5 optimization steps (first-step
   jump, then clearly diminishing per-step gains). On these 6 new targets, the
   opposite: per-step gains are flat or *increasing* through all 5 steps — no
   sign of convergence. Yet the method still found 3 hits. Is 5 steps
   systematically too few for these targets, or is under-convergence actually
   part of why the greedy accumulation works (each step commits only a small,
   local move rather than fully exploring)? (One prior experiment found that
   *more* steps, M=15, actually made the original `7txw` recipe collapse to a
   worse outcome — so "just run more steps" isn't free.)
2. **The ipTM-selection gap.** Is there a cheap, label-free confidence signal
   that would have surfaced `8urf`'s and `8vzo`'s true hits (which raw ipTM
   badly under-ranked), or are these fundamentally "found in hindsight only"
   discoveries that a real blind deployment would have thrown away?
3. **When to add intra-chain loss terms.** The pLDDT gap heuristic (§3b) is a
   cheap way to *detect* a misfolding-type failure without ground truth. Is
   there a principled way to decide *automatically*, per-target, whether to
   run the plain interface-only recipe vs. one with intra-chain terms added,
   rather than diagnosing by hand after a 12h campaign fails?
4. **z0 vs. s_lm as levers.** Is there reason to expect jointly optimizing both
   to be strictly better than either alone, or could they interfere / trade off
   against each other in a way that's not obvious from the compute-cost
   analysis?
5. **Generalization.** All of the above was tuned/discovered on a handful of
   targets. What's the right way to validate that any given adjustment (joint
   z0, intra-chain terms, different M) generalizes rather than overfitting to
   the specific 6 (now 7, counting `8vvl`'s extension) targets in hand?
