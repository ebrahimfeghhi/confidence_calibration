# Overnight, 2026-07-13/14 — what we learned, and what it costs us

_Read this first. Everything below is reproducible from a named script in `steer/experiments/`.
Where a result is underpowered or unresolved, it says so._

---

## The one-paragraph version

We set out to explain **why Protenix selects better than ESMFold2** — with the hypothesis that
AF3-class models train their confidence head as an **error discriminator** on their own decoys,
while ESMFold2's ipTM merely measures **self-consistency**.

**The headline is not the hypothesis. It is this:**

> ### The confident-decoy trap is caused by running SINGLE-SEQUENCE. An MSA cures it.
>
> ESMFold2's confidence **inverts inside its own top decile** — it discriminates well across the
> pool and then points at decoys exactly where `argmax` reads. That inversion is present in **both**
> checkpoints **without** an MSA, and **absent in both** *with* one. **Give the model an MSA and the
> disease goes away.**

And we have been running **single-sequence this entire project** — `msa_encoder.enabled = false` in
the ESMFold2-**Fast** checkpoint, and the driver even filters the MSA-bearing JSONs out. The central
phenomenon this project is built on is, at least in large part, **an artefact of a handicapped
configuration.**

The night also produced a **negative result that is more valuable than the hypothesis was**:
**a confident decoy and a confident truth are structurally indistinguishable.** That kills
structure-only label-free selection outright, and it explains why *every* method we tried — all of
them — hit the same ceiling.

**The original hypothesis is neither confirmed nor refuted.** The ablation is underpowered and its
decomposition moved a lot between 8 and 14 seeds (§3). Do not quote it.

---

## 0. TEST-TIME SCALING DOES NOT WORK ON THIS CONFIGURATION

**The repo is called `tts_abag`. Its scaling curves were measured on the one setup where scaling
does not pay.**

Full common benchmark, **148 interfaces**, seeds 1 → 50, OST-scored, raw `pair_iptm`:

| | n=1 | n=10 | **n=50** | gain |
|---|:--:|:--:|:--:|:--:|
| **coverage** — pool contains a good pose — ESMFold2-Fast | 51% | 75% | **85%** | +34% |
| **coverage** — Protenix | 58% | 79% | **89%** | +31% |
| **selector** — argmax finds it, given it exists — ESMFold2-Fast | 86% | 67% | **63%** | **−23%** |
| **selector** — Protenix | 89% | 83% | **83%** | **−7%** |
| **TOP-1** — ESMFold2-Fast | 44% | 50% | **54%** | **+11%** |
| **TOP-1** — Protenix | 51% | 66% | **74%** | **+22%** |

**Both models find the answer about equally often** (85% vs 89% coverage). **The entire difference
is that Protenix can RECOGNISE it and ESMFold2 cannot** (selector 83% vs 63%) — **and that gap
WIDENS with scale**, because ESM's selector decays three times faster.

**Test-time scaling gives ESMFold2 poses it cannot use.** Protenix converts 2× more of its scaling
into accuracy.

And on the 4-arm set (15 interfaces, 1 → 20 seeds), the worst case is stark:

| arm | top-1 @ n=1 | top-1 @ n=20 | **scaling gain** | selector decay |
|---|:--:|:--:|:--:|:--:|
| **base ESMFold2, no MSA** | 25% | **21%** | **−3%** | **−32%** |
| base ESMFold2, +MSA | 36% | 53% | +17% | −6% |
| Protenix, +MSA | 40% | 75% | +35% | −1% |

**A single-sequence ESMFold2 gets ZERO benefit from test-time scaling** — its selector decays exactly
as fast as its coverage improves. An MSA rescues the slope (−32% → −6%); Protenix's is essentially
flat (−1%).

_`scaling_by_arm.py`  (`--full` for the 148-interface version)_

---

### And the prize is growing

**HEADROOM** = `coverage × (1 − selector)` = the top-1 a *perfect* label-free selector would add,
using only poses the model **has already generated**. No extra compute; purely better recognition.

| seeds | ESMFold2-Fast: top-1 → oracle | **headroom** | Protenix: top-1 → oracle | **headroom** |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 44% → 50% | **7%** | 52% → 57% | 6% |
| 10 | 50% → 75% | **25%** | 66% → 80% | 14% |
| **50** | 54% → **86%** | **31%** | 74% → 89% | **16%** |

**A perfect selector would take ESMFold2 from 54% to 86% at 50 seeds — and the headroom GROWS with
depth (7% → 31%).** The more you spend on sampling, the more imperfect selection costs you.

**This validates the project's founding premise.** Selection *is* the bottleneck, and test-time
scaling makes it worse — so the case for a label-free selector gets **stronger** as compute grows.

The catch is §1: none of this headroom is reachable from the **pose ensemble**. It has to come from a
better confidence *signal* — an MSA (worth +38% selector on its own), a discriminator trained on
measured error, or a second model.

_`selection_headroom.py`_

---

## 0b. Is Protenix just trained on more/better data?

Ebrahim's question. **Partly yes — and I initially got this wrong** by leaning on coverage (a
saturating statistic) to claim the generators were "nearly tied". The measures that actually probe
the trunk (148 interfaces, 50 seeds):

| | ESMFold2-Fast | Protenix | gap |
|---|:--:|:--:|:--:|
| coverage *(saturating — says little)* | 84% | 89% | +5% |
| **pool purity** | 41% | 50% | **+10%** |
| **best DockQ in pool** (the ceiling) | 0.613 | 0.697 | **+0.084** |
| **median DockQ** (the *typical* pose) | 0.310 | 0.373 | **+0.063** |
| % of poses > 0.80 | 16% | 20% | +3% |
| **SELECTOR** | **65%** | **83%** | **+18%** |

**Protenix's trunk is genuinely better** — better typical pose, purer pools, higher ceiling. Better
data/training is clearly part of it.

**But the selector gap (+18%) is roughly double any generator gap (+3 to +10%).** The advantage is
disproportionately in *recognition*, not *construction*. And that matches the counterfactual: hold
each model's own confidence values, resample its pool to the other's purity, and **~2/3 of the
selector gap survives**.

**Best synthesis this data supports:** ~**1/3** of Protenix's selector advantage is a better pool
(which better data would produce); ~**2/3** points at the **confidence head** — the one component
confirmed in source to be trained differently (on the measured error of its own rollout samples).

**Standing caveat:** the *direct* measurement of head quality at the top is **not significant at
n=13** (+0.048, 95% CI [−0.506, +0.603], P = 0.56). The full re-baseline would settle it with ~150
interfaces.

_`generator_quality.py`, `residual_gap.py`, `purity_vs_head.py`_

---

## 0c. Is the "selector gap" real, or is it generation in disguise?

Ebrahim's objection, and the sharpest one of the night: the `selector` statistic
(P(argmax ipTM hits | a good pose exists)) is conditioned on coverage but is **still
prevalence-sensitive** — a purer pool makes argmax land on a good pose more often even with an
*identical* ranker, and Protenix's pools *are* purer.

So: measure ORDERING only, using statistics invariant to base rate — per-target Spearman(ipTM,
DockQ), AUC, Cliff's δ, **normalised ORACLE GAP** (rank of the first acceptable pose ÷ pool size;
this is the objection's own metric, and prevalence moves numerator and denominator together so a
richer pool cannot game it), and LIFT@k (precision@k ÷ base rate).

### Full benchmark (90 targets, 50 seeds) — NO GAP

| | Spearman ρ | AUC | oracle gap | median oracle rank |
|---|:--:|:--:|:--:|:--:|
| ESMFold2-Fast | **+0.466** | 0.770 | 9.0% | **1 of 250** |
| Protenix | +0.336 | 0.776 | 7.6% | **1 of 250** |
| P(Protenix better) | **0.002 — ESM WINS** | 0.587 | 0.722 | — |

**On the benchmark as a whole there is no selection-quality gap at all.** By Spearman, ESMFold2's
ipTM orders poses *significantly better*. **My "Protenix's confidence head is globally better"
framing was wrong** — averaging over FoldBench hides the phenomenon completely.

### Critical band (11 hard targets, pools 10–50% acceptable) — A LARGE GAP

| | Spearman ρ | **AUC** | **Cliff's δ** | **oracle gap** |
|---|:--:|:--:|:--:|:--:|
| ESMFold2-Fast | +0.337 | 0.636 | +0.271 | **10.5%** |
| base ESMFold2 + MSA | +0.263 | 0.687 | +0.374 | 2.4% |
| Protenix | +0.329 | **0.825** | **+0.651** | **1.8%** |
| P(Protenix better) | 0.485 — tie | **0.999** | **0.999** | **0.978** |

**ESMFold2 buries the first good pose at 10.5% depth; Protenix at 1.8% — six times shallower.**

### The answer

**It is a genuine selection-quality gap, not a generation artefact — but ONLY on hard targets.** The
oracle gap is immune to pool purity, and Protenix wins it decisively on the critical band and not at
all on the full benchmark.

**Consequence for every future comparison: STRATIFY BY POOL PURITY, or you will report a null.** The
phenomenon lives on ~11–30 hard targets and every full-benchmark average washes it out. This is
almost certainly why "ESMFold2's confidence is broken" and "ESMFold2's confidence is fine" can both
look true depending on which subset you examine.

_`selection_quality.py`_

---

## 1. The result that matters most

**There is no structural signature in a pose ensemble that distinguishes a confident decoy from a
confident truth.**

Depth-matched 250-pose pose-vs-pose DockQ matrices for both models (144 interfaces), restricted to
the selection-critical band (pools 10–50% acceptable — where argmax is neither trivial nor
hopeless):

| top 10% by confidence | mutual DockQ | distinct basins | share in biggest basin | **purity** |
|---|:--:|:--:|:--:|:--:|
| Protenix | 0.482 | 3.9 | 74% | **70%** |
| ESMFold2 | 0.504 | 4.8 | 69% | **51%** |

Both models pile ~70% of their most-confident poses into a single tight basin. **Same cohesion,
same basin count, same concentration. The only thing that differs is whether that basin is
correct.**

**Consequence: no clustering, no τ, no scoring rule, no cluster-mass, no HDBSCAN can work.** They
all failed for the same reason, and it was not a defect in any of them. **The information is not in
the ensemble.** We proved this the hard way — a *perfect* oracle clustering scores **3/22**, worse
than the real method's 6/22.

_`top_decile_structure.py`_

---

## 2. The mechanism (real, but narrower than we thought)

**ESMFold2-Fast's confidence discriminates across the pool and inverts inside its own top decile.**

Selection-critical band, depth-matched, raw `pair_iptm`:

| | selector | Cliff's δ, **all poses** | Cliff's δ, **top 10%** | interfaces inverted |
|---|:--:|:--:|:--:|:--:|
| Protenix | **79%** | +0.473 | **+0.477** (survives) | 11% |
| ESMFold2-Fast | **35%** | +0.404 | **−0.011** (dead) | **64%** |

`argmax` reads exactly the top. **The confidence works everywhere except where we look.**

Per-interface the correspondence is exact: every inverted top decile lost; every positive one won.

And it makes a prediction with a **control that moves the opposite way** — back off from the argmax
and it *helps* ESM, *destroys* Protenix (interaction **+40%, 95% CI [+10%, +71%], P = 0.996**).

**And §3 shows what causes it: running single-sequence.** The base checkpoint inverts too, without
an MSA — and *both* checkpoints stop inverting once given one. So this mechanism is real, and it is
a property of **MSA-less** ESM confidence, not of ESMFold2-Fast specifically.

_`critical_band.py`, `dont_take_argmax.py`, `argmax_dissociation_test.py`_

---

## 3. The MSA cures the inversion — the finding

Four arms, OST-scored, identical seeds and complexes, raw `pair_iptm`, **100-pose pools**
(20 seeds × 5 samples), 9 complexes / 15 interfaces.

`Cliff's δ TOP` = does confidence still separate good from bad poses **among the most confident
poses** — the only region `argmax` reads? Positive = healthy. **Negative = inverted = the
confident-decoy disease.**

| arm | δ, all poses | **δ, top quartile** | selector | AUC |
|---|:--:|:--:|:--:|:--:|
| **A** — ESMFold2-Fast, **no MSA** *(what this project used)* | +0.359 | **−0.139** | 40% | 0.676 |
| **A′** — base ESMFold2, **no MSA** | +0.209 | **−0.200** | 25% | 0.602 |
| **B** — base ESMFold2, **+MSA** | +0.464 | **+0.327** | **62%** | 0.728 |
| **C** — Protenix, **+MSA** | +0.623 | **+0.568** | 86% | 0.795 |

**Both no-MSA arms invert. Both with-MSA arms are healthy.** It is not the checkpoint — the *base*
model inverts too, without an MSA. **It is the MSA.** The top-quartile statistic is stable across
40-, 70- and 100-pose pools.

**And the selector follows: 25% → 62% on the same checkpoint —
+38%, 95% CI [+15%, +62%], P(MSA helps) = 0.998.** Significant.

### The rest of the decomposition — read the caveat

| component of the ESMFold2-Fast → Protenix selector gap | @ 8 seeds | @ 14 | **@ 20 (final)** |
|---|:--:|:--:|:--:|
| checkpoint (Fast → base, no MSA either way) | +17% | −19% | **−15%** |
| **MSA** (base, no MSA → base + MSA) | +13% *(n.s.)* | +31% | **+38%** *(P = 0.998)* |
| unexplained (base+MSA → Protenix) | −1% | +19% | **+18%** |

The **8-seed run was the outlier** — an earlier version of this document used it to declare the
confidence-head hypothesis dead. That was wrong. At 14 and 20 seeds the checkpoint term is
consistently **negative** (base ESMFold2 is, if anything, a slightly *worse* selector than -Fast
when neither has an MSA), and **+18% stays unexplained** between base+MSA and Protenix.

**So the confidence-head hypothesis is UNRESOLVED, not refuted.** Something still separates Protenix
from a properly-configured ESMFold2. It could be the head, or MSA depth (we capped at 512; Protenix
uses 10k–16k), or noise (n=13).

_`inversion_across_arms.py`, `msa_ablation_result.py`, `run/esm/run_msa_ablation.py`_

---

## 4. What we established about the code (facts, not inference)

- **Protenix's confidence head IS a rollout-trained error discriminator.** Mini-rollout under
  `no_grad` (`protenix.py:720-755`), head run on that sample (`:773-781`), targets are the sample's
  **real error** vs ground truth (`loss.py:1757-1795`), trunk frozen (`confidence.py:65`). Whether
  this is *why* it out-selects ESMFold2 is **unresolved** — see §3.
- **ESMFold2's head is also structure-conditioned** — takes sampled coords, re-runs per diffusion
  sample (`modeling_esmfold2.py:1061-1074`). Not a trunk read-out. My original mechanism was wrong.
- **ESMFold2-Fast has `msa_encoder.enabled = false`.** Base `biohub/ESMFold2` has it **true**. Our
  driver also filters MSA-bearing JSONs out (`run_foldbench_abag.py:46-48`). **We have been running
  single-sequence this whole time.**
- **Protenix's `ranking_score`** = `0.8·iptm + 0.2·ptm − 100·has_clash` (`disorder` is hard-coded to
  0). ESM's `ranking_score` is bit-identical to its `iptm`.
- **Templates are not the explanation** — Protenix's advantage is *larger* on the 114 zero-template
  interfaces than on the 34 with templates.

---

## 5. Does the trap class survive an MSA? Yes — but it is HALVED

The whole project targets one failure class: **TRAP** = confidence picks a pose below DockQ 0.23
while an acceptable pose sits in the pool. If MSAs dissolve that class, the method work was aimed at
a phantom. Classification on identical seeds and interfaces (20 seeds × 5 samples):

| arm | WIN | **TRAP** | NO-POSE | **trap rate** |
|---|:--:|:--:|:--:|:--:|
| **A** — ESMFold2-Fast, no MSA *(what we benchmarked)* | 6 | **9** | 0 | **60%** |
| **A′** — base ESMFold2, no MSA | 3 | **9** | 3 | **60%** |
| **B** — base ESMFold2, **+MSA** | 8 | **5** | 2 | **33%** |
| **C** — Protenix, +MSA | 12 | **2** | 1 | **13%** |

**Of the 9 traps in the no-MSA arm, the MSA cures 4 (44%) — and never damages the pool** (zero
converted to NO-POSE; the MSA fixed the *selection*, the good pose was already there).

**So the trap is real, and running single-sequence roughly DOUBLED it (60% → 33%).** Five of fifteen
interfaces stay trapped even with an MSA, and Protenix still has two. **The project has a genuine
subject — it is just half the size the benchmark said**, and every headline number in this repo (the
47-interface trap set, the 6/22) is inflated by the missing MSA.

_`traps_survive_msa.py`_

---

## 6. Remaining open questions

**1. Is anything left for the confidence head?** At 14 seeds there is +19% unexplained between
base+MSA and Protenix. That could be the head, or MSA depth (we capped at 512; Protenix uses
10k–16k), or noise (n=13). **Settle it by running base ESMFold2 with a full-depth MSA.**

**2. Powering all of the above.** Everything here rests on 9 complexes / 15 interfaces / 20 seeds.

**A correction to my own earlier claim:** I wrote that base ESMFold2 "stalls entirely on the five
largest complexes (8the, 8ts0, 8uky, 8w84, 8w90)". **That is false, and I should have caught it —
`8ts0` is 619 residues, *smaller* than several complexes that fold without trouble (8oni, 974), so
size was never a coherent explanation.**

Tested directly: **`8ts0`, `8w84` and `8w90` all fold fine — ~192s each with an MSA.** Only **`8the`
(1,230 residues) is genuinely slow: >20 minutes for a single fold**, versus ~192s for the rest. The
runner walks the complex list *in order*, so 8the — sitting 10th — blocked everything behind it, and
the other four were **never reached**. I saw "no output for 20 minutes" and inferred a five-complex
stall from evidence that only ever supported one.

**The re-baseline is therefore far more feasible than I said.** Sort the complex list by size or
shard it, and only `8the` (and possibly `8uky`, 1,204 residues, untested) costs anything real.

_`inversion_across_arms.py`_

---

## 7. What to do next

**The cheap, dull fix first.** Before any more method work: **re-run the benchmark on base
ESMFold2 with MSAs.** That is worth ~+29 points of selector accuracy — more than anything we
invented — and it costs nothing but compute. Everything else should be measured against *that*
baseline, not against ESMFold2-Fast single-sequence.

**Then, if the confident-decoy trap survives:** §1 says the answer cannot come from the pose
ensemble. It has to come from outside — a discriminator trained on measured error, an independent
epitope predictor, physics. We have **374k OST-scored ESM poses** sitting in
`dockq_ost_esmfold2fast_tts_deep.csv`, which is a ready-made training set for exactly such a
discriminator.

**And drop:** τ-tuning, HDBSCAN, cluster-mass, min-cluster-size, scoring-rule sweeps. All swept,
all capped by §1.

---

## 8. Method notes worth keeping

- **DockQ has exactly two code paths.** OST = ground truth. `dockq_torch.py` = pose-vs-pose only.
  Protenix matrices are now certified against OST (46/46 pairs ≤ 0.0005; up to 0.0045 on interfaces
  with modified residues like 8vvl's pyroglutamate).
- **gemmi 0.6.7 silently returns zero models for ESMFold2's CIFs**; 0.7.5 reads both models'.
  Building the two sides of a comparison with two different parsers is how a fake result gets made.
  Both matrix sets use `.venv-gpu` / gemmi 0.7.5.
- **Verify OST comparisons with an explicit `-c REF:MDL` chain map.** Left alone, OST re-derives its
  own correspondence and on a multi-copy assembly pairs a different copy than your matrix used —
  which manufactured a phantom 0.2018 "mismatch" until it was pinned.
