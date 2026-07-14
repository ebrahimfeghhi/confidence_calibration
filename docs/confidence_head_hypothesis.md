# Why is Protenix a better selector than ESMFold2?

_Overnight run, 2026-07-13/14. Every number below is reproducible from a named script in
`steer/experiments/`. Read `## Verdict` first if you read nothing else._

---

## The question

Label-free selection on ESMFold2 is at its ceiling. Give our clustering method a **perfect**
enumeration — every acceptable pose gathered into one hypothesis — and rank by median ipTM, and it
scores **3/22**, *worse* than the 6/22 the real method achieves. No clustering algorithm, no τ, no
scoring rule can fix that; the signal is not in ipTM to be extracted.

Protenix, on the same interfaces, selects far better. **Why?**

The hypothesis under test (H2): AF3-class models train the confidence head via a **mini-rollout** —
sample a structure, then regress the head onto *that sample's actual error* versus the ground
truth — making it a **discriminator trained on the model's own decoys**. ESMFold2's ipTM, by
contrast, was assumed to be a trunk read-out measuring **self-consistency, not correctness**.

---

## Verdict

**SUPERSEDED IN PART — read
[`overnight_2026_07_14_summary.md`](overnight_2026_07_14_summary.md) first.**

The "the confidence-head account is dead" verdict below was written from an **8-seed** run. At 14
and 20 seeds the decomposition tells a different story: the checkpoint term is consistently
**negative** and **+18% stays unexplained** between base ESMFold2+MSA and Protenix. **The hypothesis
is UNRESOLVED, not refuted.**

**What the ablation DID establish, at 100-pose pools and P = 0.998 — and it is the bigger finding:
the confident-decoy inversion is caused by running SINGLE-SEQUENCE, and an MSA cures it.**
Both checkpoints invert without an MSA (Cliff's δ at the top: −0.139, −0.200); both are healthy with
one (+0.327, +0.568). The MSA lifts the selector **25% → 62%** on the same checkpoint
(**+38%, 95% CI [+15%, +62%]**) and **halves the trap rate (60% → 33%)**.

**What survives, in order of how much it is worth:**

1. **A confident decoy and a confident truth are structurally indistinguishable** (§5). Same
   cohesion, same basin count, same concentration in the top decile — only *correctness* differs.
   **So no structure-only, label-free method can separate them**, and that is why every clustering
   approach we tried hit the same wall. This is the most valuable result here and it depends on
   none of the rest.

2. **The confident-decoy INVERSION is caused by running single-sequence, and an MSA cures it**
   (§7). Confidence discriminates across the pool and then inverts inside its own top decile —
   exactly where `argmax` reads. **Both** checkpoints invert **without** an MSA; **both** are
   healthy **with** one. We have been running single-sequence this entire project.

3. **Protenix's head IS a rollout-trained error discriminator.** Confirmed in source (§2). Whether
   that is *why* it out-selects ESMFold2 is a separate question, and it is open.

4. **ESMFold2's head is NOT a trunk read-out.** It is also structure-conditioned. My original
   mechanism was wrong (§2).

**What is UNRESOLVED (and what an earlier version of this document wrongly called settled):**

5. **Whether Protenix's confidence head explains its selector advantage.** At 14 seeds Protenix
   still out-selects base ESMFold2 + MSA by **+19%**, so something is left unexplained. It could be
   the head; it could be MSA depth (we capped at 512, Protenix uses 10k–16k); it could be noise
   (n=13). The ablation's decomposition **flipped sign between 8 and 14 seeds** — see §7 — so **do
   not quote it.**

---

## 1. Confounds killed first

Three things could have explained the gap without any claim about confidence heads. All three are
dead.

**Template leakage — ruled out.** Protenix ran with templates ON; 28 of 86 complexes in the
inference logs found 4 templates each. On the **114 zero-template interfaces** Protenix's selector
scores **83%** versus ESM's 61% — *larger* than on the 34 template interfaces (81%). Not leakage.
_(My first attempt at this join grepped backwards through the logs and mis-attributed every
template; template lines **precede** the inference line they belong to. Fixed in
`template_status()`.)_

**The composite-score confound — dead.** ESM's `ranking_score` is bit-identical to its `iptm`.
Protenix's is an AF3 composite: `0.8·iptm + 0.2·ptm + 0.5·disorder − 100·has_clash`, with
`disorder` hard-coded to zero (`sample_confidence.py:171-177`). So I had been comparing a composite
against a raw score. Scoring Protenix with **raw `pair_iptm` alone** still gives **83%**. The
ranking formula explains **none** of the gap.

**Depth — controlled.** ESM has 1,025 seeds × 5 samples = 5,125 poses; Protenix has 50 × 5 = 250.
Every comparison below subsamples ESM to 50 seeds (20 draws, same rng seed 20260713 throughout, so
the pools are literally the same poses).

_Script: `confidence_discrimination.py` → `results/csv_files/confidence_analysis.csv`_

---

## 2. What the source actually says

**Protenix — a rollout-trained error discriminator. Confirmed.**

| | evidence |
|---|---|
| mini-rollout exists | `protenix/model/protenix.py:720-755` — under `torch.no_grad()`, `sample_diffusion` for `N_step_mini_rollout=20` (`configs_base.py:183-184`), detached into `pred_dict["coordinate_mini"]` |
| head runs **on that sample** | `protenix.py:773-781` (`x_pred_coords=coordinate_mini`); it embeds the sample's pairwise distances via `torch.cdist` (`modules/confidence.py:276-306`) |
| targets are that sample's **real error** | pLDDT ← true per-atom LDDT (`loss.py:1757-1775`); PDE ← \|d_pred − d_true\| (`loss.py:1776-1782`); PAE ← real alignment error (`loss.py:1787-1795`) |
| trunk is frozen | `ConfidenceHead.stop_gradient=True` (`confidence.py:65`) |

**ESMFold2 — also structure-conditioned. H2's architectural claim is wrong.**

The model lives in the venv (`transformers/models/esmfold2/modeling_esmfold2.py`), not `~/esm`.
Its confidence head **takes the sampled coordinates as input** (`x_pred=sample_coords`, :1061-1074),
bins their pairwise distances into its own pair representation (:209-216), and **re-runs once per
diffusion sample** (:503-504). ipTM is the analytic TM-expectation over the 64-bin PAE softmax
restricted to inter-chain pairs (:279-316). It is **AF3-like, not AF2-like**.

There is **no mini-rollout** — it scores the final samples. And **what it was trained against is
not determinable**: no loss or training code ships with the HF module. So "trained on measured
error" can be neither confirmed nor refuted for ESM.

> **So the two heads differ in _what they were trained on_ (unknown for ESM; own-rollout error for
> Protenix), NOT in _whether they see the structure_ (both do).**

---

## 3. The dissociation: same average discrimination, opposite behaviour at the top

Depth-matched, raw `pair_iptm` for both, 87 zero-template interfaces where both pools contain an
acceptable pose:

| | selector (argmax) | **AUC** (average discrimination) | pool purity |
|---|:--:|:--:|:--:|
| Protenix | **83%** | 0.785 | 61% |
| ESMFold2 | **67%** | **0.797** | 53% |

**ESM's confidence orders good-vs-bad poses slightly *better* than Protenix's across the pool as a
whole — and still lands on a decoy far more often.**

AUC averages over every good/bad pair. `argmax` reads exactly one point: the maximum. A model can
be well-calibrated in bulk and catastrophic at the extreme right tail.

### Where the gap lives

Selector accuracy, stratified by pool purity:

| pool purity | Protenix | ESMFold2 | gap |
|---|:--:|:--:|:--:|
| 0–10% acceptable | 23% | 26% | −3% |
| **10–25%** | **83%** | **25%** | **+58%** |
| **25–50%** | **73%** | **45%** | **+27%** |
| 50–75% | 100% | 100% | 0 |
| 75–100% | 100% | 100% | 0 |

When the pool is mostly right, both models trivially succeed. When it is hopeless, both fail.
**The entire gap is in the band where selection actually matters** — pools that are mostly wrong but
hold a real minority of correct poses. That band *is* the confident-decoy regime.

### Head, or pool?

Counterfactual: resample each pool to the other model's purity, keeping each model's own confidence
values attached to its own poses (so the head's ability is untouched; only prevalence changes).

| | |
|---|---|
| observed gap | +16% |
| gap with purity equalised | **+11%** |
| **→ 30% is pool purity, 70% is the confidence head** | |

_Script: `purity_vs_head.py`_

---

## 4. The mechanism: ESM's confidence inverts inside its own top decile

Restricted to the selection-critical band (pools 10–50% acceptable):

| | selector | Cliff's δ, **all poses** | Cliff's δ, **top 10%** | interfaces with an inverted top decile |
|---|:--:|:--:|:--:|:--:|
| **Protenix** (n=24) | **79%** | **+0.473** | **+0.477** — survives | **11%** |
| **ESMFold2** (n=20) | **35%** | **+0.404** | **−0.011** — dead | **64%** |

Over the pool as a whole the two heads are comparable. **Inside the top 10% by confidence,
Protenix's discrimination holds and ESM's collapses to zero.** `argmax` reads precisely there.

Per-interface the correspondence is exact — every ESM interface with an inverted top decile lost,
every one with a positive top decile won:

| interface | Cliff's δ, top 10% | outcome |
|---|:--:|---|
| 8the_AB | **−0.738** | fail |
| 8w84_AD | −0.492 | fail |
| 8ts0_AC | −0.421 | fail |
| 7zoz_AB | −0.273 | fail |
| 8che_BE | **+0.632** | **win** |
| 7txw_AC | **+0.976** | **win** |
| 8qz3_AC | **+1.000** | **win** |

_Script: `critical_band.py`_

### The prediction it makes, and the control that confirms it

If ESM's very top is contaminated, **don't read the very top**. Protenix is the control: its top
decile is healthy, so backing off should only *hurt* it.

| | argmax | rank-10 | **rank-20** | rank-50 |
|---|:--:|:--:|:--:|:--:|
| **ESMFold2** | 35% | 45% | **50%** | 50% |
| **Protenix** | **79%** | 71% | **54%** | 42% |

Paired bootstrap over interfaces (10,000 resamples):

| | Δ (rank-20 − argmax) | 95% CI | P(Δ>0) |
|---|:--:|:--:|:--:|
| ESMFold2 | +15% | [−10%, +40%] | 0.836 |
| Protenix | −25% | [−42%, −8%] | 0.000 |
| **interaction** | **+40%** | **[+10%, +71%]** | **0.996** |

**Honest reading.** ESM's own +15% gain is **not significant** (n=20 — three interfaces). I cannot
claim "backing off fixes ESMFold2". What *is* established is the **contrast**: the two models
respond to ignoring their own top-ranked pose in **opposite directions**, P = 0.996. A control that
moves the other way is far harder to obtain by chance than a bare effect, which is why the
interaction is the claim worth making.

_Scripts: `dont_take_argmax.py`, `argmax_dissociation_test.py`_

---

## 5. The negative result that matters most

**Why** is ESM's top decile contaminated? The project's standing law says the confidence head
measures self-consistency, and a rigid wrong-register dock is maximally self-consistent. That
predicts ESM's top decile should **collapse onto one tight structural basin** while Protenix's —
trained on error — stays diverse.

Tested on depth-matched 250-pose pose-vs-pose DockQ matrices for **both** models (144 interfaces),
selection-critical band:

| top 10% by confidence | mutual DockQ | distinct basins | share in biggest basin | **purity** |
|---|:--:|:--:|:--:|:--:|
| Protenix | 0.482 | 3.9 | 74% | **70%** |
| ESMFold2 | 0.504 | 4.8 | 69% | **51%** |

**The prediction fails.** The two top deciles are **structurally indistinguishable** — same
cohesion, same basin count, same concentration. Both models pile ~70% of their most-confident poses
into a single tight basin. **The only thing that differs is whether that basin is correct.**

> ### This is the important finding.
>
> **There is no structural signature in the pose ensemble that distinguishes a confident decoy from
> a confident truth.** A wrong basin and a right basin look the same: equally tight, equally
> reproducible, equally dominant.
>
> **Therefore no structure-only, label-free selector can work** — not clustering, not τ-tuning, not
> HDBSCAN, not cluster-mass, not any of the estimators we swept. They all failed for the same
> reason, and it was not a defect in any of them. The information simply is not in the ensemble.
>
> The information that separates the two models lives **inside Protenix's confidence head** —
> which was trained, on ground truth, to predict its own error.

_Script: `top_decile_structure.py`_

---

## What this changes

- **Stop trying to fix selection with the pose ensemble.** The ceiling is not an artefact of our
  clustering; it is a property of the data. The ensemble does not know.
- **A discriminator trained on error is the lever.** Protenix's head is exactly that, and it is
  the only thing in this study that separates truth from confident decoy in the critical band.
- **The cheapest available win is cross-model:** use Protenix's error-trained head. It is already
  on disk and it is right where ESM is wrong.
- **A trainable direction, if we want one:** fine-tune a confidence head on ESMFold2's own decoys
  against measured DockQ — the mini-rollout recipe, applied to ESM. We have 374k scored ESM poses
  sitting in `dockq_ost_esmfold2fast_tts_deep.csv`. That is a *training set for exactly this*.

---

## Caveats

- **ESM's training targets are unknown.** No training code ships. The claim "Protenix's head is a
  discriminator and ESM's is not" is established for Protenix and **only inferred** for ESM, from
  behaviour rather than source.
- **n is small in the critical band** (20 ESM / 24 Protenix interfaces). The *interaction* is
  significant; the individual effects are not.
- **DockQ matrices**: certified against OST at 46/46 pose pairs across 12 random interfaces
  (worst diff 0.0005 = OST's rounding). Known limit: interfaces carrying modified residues
  (e.g. 8vvl_AC, pyroglutamate) deviate by up to 0.0045 — negligible for cohesion/clustering, but
  it is not zero. See `verify_protenix_dockq.py`.
- **The MSA confound.** ESMFold2-Fast runs **single-sequence** — `msa_encoder.enabled = false` in
  the checkpoint we use, while base `biohub/ESMFold2` has it **true**, and Protenix uses MSAs. So
  the comparison has been **MSA vs no-MSA** all along, and that could explain part or all of the
  gap. It is the largest threat to everything above. See §6.

---

## 6. The MSA confound

### 6a. MSA *depth* does not drive the advantage

If MSAs cause the gap, Protenix's advantage should **scale with MSA depth**. Protenix logs `N_msa`
per structure. On 53 zero-template interfaces where both pools contain an acceptable pose:

| Protenix MSA depth | Protenix | ESM | **advantage** |
|---|:--:|:--:|:--:|
| Q1 shallowest (10.3k–12.6k) | 87% | 60% | **+27%** |
| Q2 (12.6k–13.4k) | 92% | 58% | **+33%** |
| Q3 (13.4k–14.4k) | 69% | 85% | **−15%** |
| Q4 deepest (14.4k–16.4k) | 77% | 92% | **−15%** |

**Spearman = −0.299, p = 0.030.** The correlation runs **backwards**: Protenix's advantage is
*largest* where its MSA is *shallowest*, and it **loses** to ESM where the MSA is deepest. That is
the opposite of what the MSA hypothesis predicts.

**But this does not settle it, and it would be dishonest to pretend it does.** The **minimum** MSA
depth here is **10,262 sequences** (and `N_msa` caps at 16,384), so this compares *deep against
deeper*, in a regime where the coevolutionary signal is almost certainly saturated. The contrast
that matters is **0 vs 10,262** — ESMFold2-Fast has **no MSA at all** — and that contrast has zero
variance in this data. There is also a confound of the confound: deep MSAs mean well-studied,
conserved antigens, which are easier for *everyone*, including ESM's language model.

_Script: `msa_confound.py`_

### 6b. The ablation

The only thing that settles it: run **base `biohub/ESMFold2`** (`msa_encoder.enabled = true`) with
**Protenix's own MSAs**, on the 14 complexes of the selection-critical band, holding everything else
fixed (same TTS settings, same 5 diffusion samples, depth-matched seeds, scored by the same OST).

| outcome | meaning |
|---|---|
| ESM-with-MSA jumps to Protenix's selector accuracy | **the confidence-head account is dead** — it was MSAs |
| it stays at the single-sequence level | **MSAs are not the explanation**; the account survives |
| it lands between | both contribute; report the split |

_Scripts: `run/esm/run_msa_ablation.py`, `run/esm/launch_msa_ablation.sh`,
`steer/experiments/msa_ablation_result.py`_

---

## 7. The ablation result — and it refutes §1–§4's explanation

**A design flaw I nearly shipped.** My first ablation ran base `biohub/ESMFold2` **with** MSAs and
compared it to ESMFold2-**Fast**. But those are **different checkpoints** (1.3 GB vs 721 MB), so
that contrast varies the MSA *and* the model, and nothing could be attributed to the MSA. The fix
is a control arm: **the same base checkpoint with no MSA.**

Four arms, all scored by OST, all ranked by raw `pair_iptm`, all on the same 8 seeds × 5 diffusion
samples, on the 9 critical-band complexes that base ESMFold2 can actually fold:

| arm | selector | generator | pool purity | AUC |
|---|:--:|:--:|:--:|:--:|
| **A** — ESMFold2-**Fast**, no MSA *(what this whole study used)* | **29%** | 93% | 21% | 0.619 |
| **A′** — base ESMFold2, no MSA *(**the control**)* | **45%** | 73% | 21% | 0.655 |
| **B** — base ESMFold2, **with MSA** *(**the ablation**)* | **58%** | 80% | 29% | 0.776 |
| **C** — Protenix, with MSA | **57%** | 93% | 39% | 0.713 |

### The decomposition — AND WHY YOU MUST NOT QUOTE IT

At **8 seeds** the decomposition said: checkpoint +17%, MSA +13% (n.s.), unexplained −1% — i.e.
"nothing left for the confidence head". **At 14 seeds it says something different:**

| component of the ESMFold2-Fast → Protenix selector gap | @ 8 seeds | **@ 14 seeds** |
|---|:--:|:--:|
| **checkpoint** (Fast → base, no MSA either way) | +17% | **−19%** |
| **MSA** (base, no MSA → base, with MSA) | +13% *(n.s.)* | **+31%** *(P = 0.945)* |
| **unexplained** (base+MSA → Protenix) | −1% | **+19%** |

**The checkpoint effect flipped sign. It is noise.** And the "unexplained" term went from nothing to
+19%, so the confidence-head hypothesis is **not refuted** — merely unresolved. An earlier version
of this document declared it dead on the strength of the 8-seed run. That was wrong, and it is
exactly the failure mode this project keeps having: a confident causal story from an underpowered
measurement.

### What IS stable — and it is the finding

**The MSA effect.** It survives both depths, and both the quartile and decile views of the top-end
statistic:

| arm | δ, all poses | **δ, top quartile** | **δ, top decile** | interfaces inverted |
|---|:--:|:--:|:--:|:--:|
| ESMFold2-Fast, **no MSA** | +0.335 | **−0.070** | **−0.120** | 60% |
| base ESMFold2, **no MSA** | +0.273 | **−0.033** | **−0.333** | 71% |
| base ESMFold2, **+MSA** | +0.469 | **+0.286** | **+0.252** | 40% |
| Protenix, **+MSA** | +0.584 | **+0.542** | **+0.600** | 22% |

**Both no-MSA arms invert. Both with-MSA arms are healthy.** It is not the checkpoint — the *base*
model inverts too, without an MSA. **The confident-decoy inversion is an MSA-less phenomenon, and an
MSA cures it.** Selector follows: **27% → 58%** on the same checkpoint
(**+31%, 95% CI [0%, +62%], P = 0.945**).

### Limits of this experiment

- **9 complexes, 15 interfaces, 8 seeds.** Base ESMFold2 stalls on the five largest complexes
  (8the, 8ts0, 8uky, 8w84, 8w90 — 20+ min at 100% GPU with no output), so they are excluded from
  **all four arms**. The comparison is internally controlled but the sample is small.
- **MSA depth capped at 512**, not Protenix's 10k–16k. At the processor default of 1024 the
  with-MSA arm produced ~0 folds/hour. 512 is still a real MSA (AF2-class models subsample to
  128–512 per pass) but it is a weaker treatment than Protenix gets.
- Both arms use identical seeds per interface (intersected), so a difference can never be an
  artefact of one arm finishing the easier seeds.

### What this means for the project

- **The inversion (§4) is still real** — it is a measured fact about ESMFold2-Fast, and it explains
  why *our* clustering methods failed on *our* pools.
- **The structural negative result (§5) is still the most valuable finding here** and does not
  depend on any of this: a confident decoy and a confident truth are structurally
  indistinguishable, so no structure-only label-free selector can separate them.
- **But "we need a discriminator-trained confidence head" is the wrong lesson.** The cheap fix is
  much duller: **use the base checkpoint, and give it an MSA.** That recovers Protenix-level
  selection from ESMFold2 itself.
- **Everything in this project was measured on ESMFold2-Fast, single-sequence.** That is a
  handicapped configuration, and the "confident-decoy" phenomenon should be re-measured on base
  ESMFold2 + MSA before any of it is written up as a property of structure predictors in general.

_Scripts: `run/esm/run_msa_ablation.py`, `steer/experiments/msa_ablation_result.py`_
