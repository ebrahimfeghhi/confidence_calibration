# Overnight, 2026-07-13/14 — what we learned, and what it costs us

_Read this first. Everything below is reproducible from a named script in `steer/experiments/`.
Where a result is underpowered or unresolved, it says so._

---

## The one-paragraph version

We set out to explain **why Protenix selects better than ESMFold2** — with the hypothesis that
AF3-class models train their confidence head as an **error discriminator** on their own decoys,
while ESMFold2's ipTM merely measures **self-consistency**.

**The hypothesis is refuted by its own control.** Run the ablation properly and the entire
Protenix-vs-ESMFold2 selector gap decomposes into **the checkpoint** and **the MSA**, with
**nothing left over** for the confidence head. We spent this project on **ESMFold2-Fast**, a
distilled model, run **single-sequence** — and mistook its handicap for Protenix's virtue.

But the night also produced a **negative result that is more valuable than the hypothesis was**:
**a confident decoy and a confident truth are structurally indistinguishable.** That kills
structure-only label-free selection outright, and it explains why *every* method we tried — all of
them — hit the same ceiling.

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

**Caveat:** this is a fact about **ESMFold2-Fast**. Whether it survives on the base checkpoint with
an MSA is **unresolved** (§5).

_`critical_band.py`, `dont_take_argmax.py`, `argmax_dissociation_test.py`_

---

## 3. The refutation

Four arms, OST-scored, same seeds, same 9 complexes, raw `pair_iptm`:

| arm | selector | generator | AUC |
|---|:--:|:--:|:--:|
| **A** — ESMFold2-**Fast**, no MSA *(what this whole project used)* | **29%** | 93% | 0.619 |
| **A′** — base ESMFold2, no MSA *(the control)* | **45%** | 73% | 0.655 |
| **B** — base ESMFold2, **with MSA** | **58%** | 80% | 0.776 |
| **C** — Protenix, with MSA | **57%** | 93% | 0.713 |

| decomposition of the +29% gap | |
|---|:--:|
| **checkpoint** (Fast → base) | **+17%** |
| **MSA** (base, no MSA → base + MSA) | **+13%** *(CI [−23%, +54%], n.s.)* |
| **unexplained** | **−1%** |

**Base ESMFold2 with an MSA selects as well as Protenix (58% vs 57%).** Protenix's advantage is not
a property of a rollout-trained confidence head — an ESM-class model with an MSA matches it.

**Honesty:** the MSA effect alone is **not significant** (n=13). What *is* solid and needs no MSA
arm is the **checkpoint effect** (+17%, no MSA on either side), and that alone breaks the story.

_`msa_ablation_result.py`, `run/esm/run_msa_ablation.py`_

---

## 4. What we established about the code (facts, not inference)

- **Protenix's confidence head IS a rollout-trained error discriminator.** Mini-rollout under
  `no_grad` (`protenix.py:720-755`), head run on that sample (`:773-781`), targets are the sample's
  **real error** vs ground truth (`loss.py:1757-1795`), trunk frozen (`confidence.py:65`). This is
  true — it just isn't *why* it wins.
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

## 5. The open question that now matters most

**Does the confident-decoy phenomenon survive on base ESMFold2 with an MSA?**

Everything this project has measured was on **ESMFold2-Fast, single-sequence** — a handicapped
configuration. If the inversion vanishes once the model is properly configured, then the central
phenomenon is an artefact of our setup, not a property of structure predictors.

**We tried to answer this and could not.** At 8 seeds (40-pose pools) a "top decile" is four poses,
and Cliff's δ on four points is noise — the quartile and decile views contradicted each other
outright. Reporting either would have been dishonest. The one thing that *is* estimable at that
depth: **the MSA lifts bulk discrimination from +0.31 to +0.55, matching Protenix.**

**The experiment that answers it:** base ESMFold2 + MSA, **50 seeds × 5 samples = 250-pose pools**,
on the critical-band complexes, then rerun `critical_band.py`. That is ~2× what we ran tonight.
Note that base ESMFold2 **stalls entirely on the five largest complexes** (8the, 8ts0, 8uky, 8w84,
8w90 — 20+ min at 100% GPU with no output), so those need a different approach or exclusion.

_`inversion_across_arms.py`_

---

## 6. What to do next

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

## 7. Method notes worth keeping

- **DockQ has exactly two code paths.** OST = ground truth. `dockq_torch.py` = pose-vs-pose only.
  Protenix matrices are now certified against OST (46/46 pairs ≤ 0.0005; up to 0.0045 on interfaces
  with modified residues like 8vvl's pyroglutamate).
- **gemmi 0.6.7 silently returns zero models for ESMFold2's CIFs**; 0.7.5 reads both models'.
  Building the two sides of a comparison with two different parsers is how a fake result gets made.
  Both matrix sets use `.venv-gpu` / gemmi 0.7.5.
- **Verify OST comparisons with an explicit `-c REF:MDL` chain map.** Left alone, OST re-derives its
  own correspondence and on a multi-copy assembly pairs a different copy than your matrix used —
  which manufactured a phantom 0.2018 "mismatch" until it was pinned.
