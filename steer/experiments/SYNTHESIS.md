# Directed diversity for antibody–antigen docking — synthesis (7txw)

Status: working method validated on 7txw; hyperparameter generality and a compute-matched
comparison to the baseline are still open. Self-contained summary as of this session.

Notation: "G" = the good (native) antigen-orientation basin; "B1" = the model's default (wrong)
orientation; "B2" = a confident wrong-register decoy near B1. "wg" = cross-block distogram
W1 distance to the G centroid; real-G = wg < 0.08. "run" = one steering attempt; "trial" = one
independent chain of runs.

---

## 1. The baseline we want to beat: the ESMC masking + dropout lottery

ESMFold2-Fast predicts the complex, but for 7txw its **default prediction is the wrong antigen
orientation (B1)**; the true pose (G) is a minority outcome. The existing way to find G:

- Run many **independent seeds**, each with (a) random **ESMC input masking** (mask ~10% of the
  antigen LM tokens) and (b) **LM dropout** (rate 0.3, resampled each recycling loop). Each seed
  is a different random perturbation of the conditioning → a different orientation outcome.
- Outcome distribution over 60 labeled seeds: **~68% B1, ~22% B2, ~7% G.** So one seed hits the
  true pose ~7% of the time → ~15 seeds to a first G on average, and thousands to cover reliably.
- Selection is the easy part: **best-pair ipTM ranks G at the top at scale** (AUC ~0.945).
- It is an **undirected lottery**: coverage is the bottleneck (you must oversample), and every
  seed pays a full ESMC forward (6B params) + trunk + diffusion decode. **Without masking
  (z0 variation alone) the G rate is 0%** — masking is what creates orientation diversity; G
  specifically needs masking the epitope-adjacent antigen residues (C48–C50, found post-hoc).

**Goal:** replace the blind lottery with a *directed* search that reaches G in fewer, more
reliable attempts, ideally without per-attempt ESMC.

---

## 2. Our method: sequential greedy self-repulsion with anchor accumulation

Fix one ESMC rep (the no-mask baseline) and steer only the single LM rep `s_lm` by gradient
(no new ESMC per attempt). Per **trial**:

1. Seed a growing **anchor list** with the baseline's distogram (the B1 floor).
2. Each **run**: reset `s_lm` to baseline + a small random nudge → take **M gradient steps**
   maximizing cross-block distogram **W1 distance from every accumulated anchor** (at a short
   horizon **K**) → append the converged distogram to the anchor list.

The repulsion is **label-free** — anchors are self-generated; no G/B1/B2 knowledge enters the
steering (basin centroids are used only to *score*). Defaults: **M=5, K=3, lr=0.3, nudge=5% of
‖s_lm‖, fixed z0.**

**Key mechanism — anchor accumulation = search by elimination.** A single repel from fixed
points just shoves you into the nearest deep attractor (B2) every time — there is no memory. The
growing anchor list *is* the memory: each visited basin becomes a no-go, so the strong sinks get
crossed off one at a time (B1 → B2 → lesser decoys) until **G is the only place left**. That is
why G appears at runs 4–9 and never run 1. Accumulation is also **self-calibrating**: it
auto-scales to however many basins a pair has (a pair with more decoys just needs more runs).

This was confirmed by its failure mode: a one-shot repel from fixed B1+B2 centroids (no
accumulation) reaches G essentially never and lands in B2 — accumulation is the load-bearing piece.

---

## 3. Results on 7txw

Each config = 3 trials × 10 runs (the two M=12 probes = 2 trials × 8 runs).

| config | real-G / runs | trials-with-G | runs-to-first-G | notes |
|---|---|---|---|---|
| **nudge + fixed z0** (the recipe) | **6 / 30** | **3 / 3** | 6, 4, 9 (avg **6.3**) | best wg 0.049 |
| nudge + vary z0 | 5 / 30 | 3 / 3 | — | z0 variation neutral |
| maskreps + fixed z0 | 0 / 30 | 0 / 3 | — | masked-rep seeding fails |
| maskreps + vary z0 | 1 / 30 | 1 / 3 | — | "" |
| M=12, K=3 | 1 / 16 | 1 / 2 | — | more steps ≈ worse (drift) |

**Headline:** the plain recipe hits a real-G **within 10 runs in every trial (3/3)**, the
simplest config wins. Caveat: trial 2 hit at run 9 — consistency is real but the variance
(4 → 9) leaves little margin under a strict 10-run budget.

What helps / doesn't:
- **z0 variation: neutral** (also 3/3, no better).
- **Masked-rep seeding: fails** (0/3). Starting from a committed B1/B2 mask is too far from the
  plastic baseline to repel out of (an "init-magnitude too large" failure).
- **Steps (M): M=5 is the loss-curve knee.** Per-step repel gain is large for steps 1–3
  (+0.17/+0.09/+0.05) then flat (~+0.02/step), while off-manifold drift grows ~linearly. M=12 was
  no better and slightly worse (more drift, fewer real-G). So more steps buy ~no spread + more
  junk.
- **Nudge magnitude: looks robust.** Early sweep has 2%, 10%, and 20% all landing G — not
  knife-edge (good for generality). Final dose-response pending.
- **Horizon (K): the live lever.** K=8 grazed G at run 1 and keeps strong G-affinity (more
  runs land in/near G than K=3), suggesting **longer horizon reaches G in fewer runs** — but
  K=8 is ~5× slower per run. K-ladder (3/5/8) finishing.

---

## 4. Hyperparameter tuning — the central challenge

The recipe has constants tuned on 7txw (K=3, M=5, lr=0.3, nudge=5%, R=10). Whether they
generalize is the main open risk. Status by knob:

- **M (steps) — effectively solved.** Stop at the repel-loss plateau (validated: gains flatten by
  step ~3–4; more steps only add off-manifold drift). Self-calibrating, no magic number needed.
- **R (runs) — self-calibrating** via accumulation: "loop-until-dry" (stop when N runs find no new
  basin) auto-scales to the basin count.
- **Nudge magnitude — harder, but looks forgiving.** Too small → gradient ~0 at your own anchor
  (no diversity); too large → you start inside a committed decoy (the maskreps failure). 5% works,
  and 2–20% all reach G so far. Generalizable rule: set it relative to the pair's natural
  mask-induced `s_lm` spread (measurable from a few masks on any pair).
- **Horizon K — hardest.** It sets where the gradient "sees" the basins. K=3 is *before* the
  basins differentiate (works for `s_lm` only because `s_lm` is re-injected every loop, but
  reaches G late); full-horizon goes flat and fails; K=8 suggests longer is fine/better for the
  `s_lm` lever. Generalizable idea: derive K from the pair's differentiation onset (when basins
  separate in a baseline run), rather than fixing it.

Two deeper challenges, orthogonal to the knobs:

- **n = 1 pair.** All validation is on 7txw. The only real test of generality is running the
  identical pipeline on a held-out ab-ag complex with no retuning. Untested.
- **Label-free recognition of G.** On a novel pair there is no G centroid, so selection falls to
  ipTM — which has the B2 confident-decoy trap (high ipTM, wrong dock). "Exit timing" (G leaves
  B1 early and far) is a promising label-free handle but is itself unproven (n=1).

---

## 5. Other key details

- **Basins are antigen orientations, not refolds** (validated: B1 vs B2 differ ~6–9× more
  inter-chain than intra-chain). This justifies a label-free validity gate: a real escape
  preserves the intra-chain folds and only changes the cross-block (orientation); junk refolds.
- **B2 is the main obstacle:** a confident *wrong-register* dock, geometrically near B1, and a
  strong **late-committing sink** (commits ~loop 14). z0 variation alone reaches B1 and B2
  (~25% B2) but never G — only the conditioning lever (mask / `s_lm`) reaches G. The whole
  difficulty of the method is getting *past* B2, which is exactly what anchor accumulation does.
- **Compute honesty — beating the lottery is not yet established on raw FLOPs.** ~6.3 runs ×
  (5 grad steps at K=3 + a 17-loop eval) is the same order as the lottery's ~15 seeds ×
  (ESMC + 17-loop + decode). The method's clear wins are **directedness/reliability** (3/3 vs a
  stochastic, high-variance lottery) and **reusing one ESMC rep** (no per-attempt 6B-param
  forward); the per-run gradient cost offsets some of that. A clean compute-matched comparison is
  still to do.
- **Engineering.** The method is a sequential single-leaf gradient search (~25 GB, cheap). Runs
  are ~3 min at K=3, ~15 min at K=8 (no fused attention installed — xformers would speed this a
  lot). A separate forward-only "beam" idea maxes at ~16 particles on a 96 GB card (~5 GB each).
- **Provenance.** Faithfully reimplemented from the archived `steer_repel.py` (which produced the
  original "12 real-G over 10 trials" figure) onto `confornet/forward_grad`; the reimplementation
  reproduces G, confirming both the original result and the diagnosis that earlier one-shot
  attempts failed purely for lack of anchor accumulation.

---

## 6. Variant under test (overnight): ConforNet affine lever

Open question (from the ConforNet line of work): **can a *global* channel-wise affine on the pair
conditioning move a discrete docking pose**, instead of steering the per-residue `s_lm`? We keep
the *exact same* self-repulsion loss + anchor accumulation and only swap the lever:

- **`s_lm` lever (validated):** optimize the per-residue single rep directly (~L×256 dof). Reaches
  G 3/3 trials.
- **affine lever (ConforNet, under test):** optimize a channel-wise affine `lm_z → lm_z·Wᵀ + b`
  (W: 256×256 init identity, b: 256 init 0), applied to the fixed baseline conditioning and
  broadcast over all pairs. Much lower-dimensional and geometry-preserving — but possibly *too*
  constrained to produce a localized reorientation.

Implemented as `repel_proven.py --lever affine` (the `s_lm` path is byte-identical to the
validated recipe; affine uses grad-clipping + a symmetry-breaking 0.01 init noise on W/b).
Running overnight: `affine_lr02` (lr 0.02) and `affine_lr10` (lr 0.1), M=12, K=3, **5 trials** × 10
runs. Result pending. If the affine reaches G consistently, it's a smaller, cleaner lever to
prefer; if not, it bounds the expressivity a global transform has for this problem.

---

Code: `analysis/beam_probe/repel_proven.py` (configurable: `--lever {slm,affine} --maskreps
--zvary --init-frac --M --K --lr --trials --runs --tag`; per-step loss + drift logged to
`repel_<tag>_steps.csv`). Daily worklog: `docs/worklog_2026-06-25.md`.
