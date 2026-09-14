# Why test-time seed scaling improves antibody–antigen docking in Protenix

**Question.** Running more random seeds substantially improves antibody–antigen structure
prediction. Across 161 labelled interfaces, going from one seed to 50 raises oracle DockQ from
0.426 to 0.697, and the pass rate of the *selected* pose (highest interface pair-ipTM) from 50.0%
to 72.7%. Something varies from seed to seed. This report identifies what.

Two hypotheses were tested:

- **H1 — fold quality.** Some seeds build a more accurate antibody paratope (CDR H3) or antigen
  epitope, and those become the good seeds.
- **H2 — epitope sampling.** Fold quality is essentially constant across seeds; what varies is
  *where on the antigen the antibody docks*, so scaling is a search over binding sites.

**Answer: H2, decisively.** Neither side of the interface is folded better in the good seeds. What
changes is which antigen surface the model commits to, and that commitment is made in the trunk.

---

## 1. Dataset and protocol

| | |
|---|---|
| Model | Protenix (protenix-v2), 10 recycles, MSA subsampling ON, MC dropout ON |
| Complexes | 104 antibody–antigen complexes with predictions and a FoldBench ground truth |
| Interfaces | 161 labelled antibody–antigen interfaces (55 complexes contribute two) |
| Depth | 50 seeds × 5 diffusion samples = 250 poses per complex, complete for all 104 |
| Total | 40,249 scored poses |

Complexes are the unit for antibody-fold measures (CDR H3 belongs to the heavy chain, one per
complex); interfaces are the unit for anything docking-related.

**All structural quantities come from OpenStructure (OST).** DockQ and its CAPRI components
(fnat, iRMSD, LRMSD) are read from OST's own output; chain correspondence between model and ground
truth uses OST's `chain_mapping`, taken **per pose** because in multi-copy assemblies OST re-picks
which copy to align against for each pose. Superposition uses `ost.mol.alg.SuperposeSVD`.

**Metrics computed:**

- **CDR H3 RMSD**, local and global. Chothia numbering via ANARCI; Cα only. *Local* superposes the
  H3 loop onto itself (loop conformation alone); *global* superposes the heavy-chain framework and
  then measures H3 (conformation plus hinge/placement). This follows Hitawala & Gray, *mAbs* 2025.
- **Epitope RMSD**, local and global. Epitope defined as antigen residues within 5 Å of the
  antibody **in the ground truth** — a fixed residue set per interface, so poses that dock elsewhere
  are still scored on the same patch. *Local* superposes the patch onto itself; *global* superposes
  the whole antigen chain then measures the patch.
- **DockQ components**: fnat (fraction of native contacts recovered), iRMSD, LRMSD.
- **Epitope clustering**: each pose's contact set (antigen residues within 5 Å of the antibody in
  the *prediction*), clustered by Jaccard ≥ 0.5 with single linkage. Single linkage is deliberately
  conservative — a gradually drifting placement stays one cluster — so the method under-counts
  distinct surfaces rather than inventing them. Clusters holding ≥5% of poses are counted.

---

## 2. Fold quality does not vary across seeds

Variance decomposition of each fold measure:

| measure | between complexes/interfaces | within (across seeds) | median within-target IQR |
|---|---|---|---|
| CDR H3 local RMSD | 84% | 16% | 0.18 Å |
| epitope local RMSD | 95% | 5% | 0.082 Å |

Fold accuracy is overwhelmingly a property of the **target**, not the seed. A typical complex builds
essentially the identical loop and the identical epitope patch in all 250 poses. For 32% of
complexes the H3 spread is below 0.10 Å — the loop is frozen.

Of the small within-complex H3 variance, 77% is seed-to-seed and 23% sample-to-sample, consistent
with the loop being determined in the trunk (a seed's five diffusion samples share one trunk state).

---

## 3. Fold quality does not predict the scaling gain

For each interface: improvement from 1 seed to 50 seeds, signed so positive = improvement, versus the
oracle DockQ gain. "1 seed" is the expectation over the 50 seeds; "50 seeds" is the best achieved.

| measure | n | Pearson r | r² | partial r, controlling starting DockQ |
|---|---|---|---|---|
| CDR H3 local | 161 | +0.181 | 0.033 | **+0.070** (p = 0.38) |
| CDR H3 global | 161 | +0.095 | 0.009 | **−0.049** (p = 0.54) |
| epitope local | 121 | −0.120 | 0.014 | **−0.238** (p = 0.009) |
| epitope global | 121 | −0.111 | 0.012 | **−0.280** (p = 0.002) |

**The control matters and is not optional.** DockQ is bounded at 1.0, so gain is arithmetically
capped by (1 − starting DockQ); empirically starting DockQ versus gain is r = −0.50, the strongest
relationship in the dataset. Hard targets have both more headroom *and* floppier loops, which
manufactures a positive correlation with no causal content. Controlling for starting DockQ removes
it: CDR H3 local falls from +0.181 to +0.070.

No fold measure explains more than 3% of the variance in scaling gain. **H1 is rejected on all four
measures, on both sides of the interface, in both local and global form.**

---

## 4. What does predict the gain: contact recovery

Same analysis on the three CAPRI components of DockQ, using the components **of the winning pose** —
i.e. when scaling yields a better final answer, what is better about that answer?

| component improvement | Pearson r | r² | partial r, controlling starting DockQ |
|---|---|---|---|
| **fnat increase** | **+0.935** | **0.874** | **+0.917** |
| iRMSD reduction | +0.686 | 0.471 | +0.565 |
| LRMSD reduction | +0.602 | 0.363 | +0.453 |

*Interpretive caveat.* DockQ is computed from these three, so all three were guaranteed to correlate
with DockQ gain; this is an attribution, not a causal test. What is **not** guaranteed is the
ordering. fnat explains 87% of the variance against 47% and 36% for the geometric terms, and
uniquely it barely moves under the headroom control. The geometric improvements are largely
downstream: land on the right surface and interface geometry improves automatically.

Consistent with this, fnat has by far the highest seed-to-seed coefficient of variation within an
interface (0.81, versus 0.45 for iRMSD and 0.39 for LRMSD).

---

## 5. Failures are wrong-surface, not bad geometry

Of 20,570 poses with DockQ < 0.23:

| failure type | n | share |
|---|---|---|
| fnat < 0.10 — **contacts essentially none of the true epitope** | 17,905 | **87.0%** |
| fnat 0.10–0.30 — partial overlap | 2,519 | 12.2% |
| fnat ≥ 0.30 — found the epitope but still failed | 146 | **0.7%** |

Failure is almost never "right place, built badly." Conversely, finding the epitope is very nearly
sufficient: once fnat ≥ 0.30, an acceptable dock follows in 99.3% of cases. "Right surface, wrong
register" accounts for only 3.7% of poses that found the epitope.

CAPRI classification over all poses: 22.6% high, 16.2% medium, 9.4% acceptable, 51.8% incorrect.

---

## 6. The model commits to very few surfaces

Clustering all 250 poses per interface by contact set (161 interfaces):

| distinct surfaces (≥5% of poses) | interfaces |
|---|---|
| 1 | 93 (58%) |
| 2 | 40 (25%) |
| 3 | 13 (8%) |
| 4–5 | 8 (5%) |
| 0 (never contacts the antigen) | 7 (4%) |

Median **1**; 95% use three or fewer. The dominant surface holds a median **79%** of all poses. The
model does not search the antigen surface — it commits to one or two candidate patches and redraws
them. Conditional on outcome (142 interfaces with a defined native epitope):

| outcome | n | median surfaces | median dominance |
|---|---|---|---|
| usually finds the native epitope | 53 | 1 | 100% |
| finds it rarely (<50% of poses) | 54 | 2 | 66% |
| never finds it | 35 | 2 | 60% |

Interfaces that succeed have locked onto the correct single surface. Interfaces where scaling pays
have **two** surfaces at roughly 66/34, with the correct one in the minority.

---

## 7. The epitope is chosen in the trunk

A seed's five diffusion samples share one trunk state. Restricting to the 58 interfaces that have
≥2 distinct surfaces (i.e. where there is a choice to make) and comparing against a permutation null:

| | value |
|---|---|
| observed: seeds whose 5 samples all land on one surface | **25.0%** |
| expected under random assignment | 7.2% |
| ratio | **3.47×** |
| interfaces where observed > null | 45/58 (78%) |
| Wilcoxon | p = 1.7 × 10⁻¹⁰ |

The epitope decision is made during recycling, not during diffusion. Seed-to-seed diversity therefore
originates in trunk-level stochasticity — MSA subsampling and dropout — which is consistent with the
prior observation in this project that running with full MSA and no dropout does not reproduce the
test-time-scaling curve.

---

## 8. Where scaling pays, with the confound removed

Oracle DockQ gain by how often the epitope is found:

| findability | n | oracle@1 | oracle@50 | gain |
|---|---|---|---|---|
| never (0%) | 18 | 0.051 | 0.111 | +0.061 |
| rare (0–10%) | 36 | 0.126 | 0.567 | **+0.441** |
| sometimes (10–50%) | 35 | 0.300 | 0.761 | **+0.460** |
| usually (50–90%) | 24 | 0.596 | 0.845 | +0.249 |
| almost always (>90%) | 48 | 0.782 | 0.885 | +0.104 |

An inverted U: no headroom at the top, nothing to find at the bottom.

Epitope-structure predictors of gain, with and without the headroom control:

| property | raw r | partial r, controlling starting DockQ |
|---|---|---|
| number of distinct surfaces | +0.331 | **+0.218** (p = 0.006) |
| dominance of the largest surface | −0.348 | **−0.270** (p = 0.0005) |
| % of poses finding the epitope | −0.468 | −0.076 (p = 0.34) — does not survive |

Matching on starting DockQ and splitting each band into high- and low-gain halves localises the
effect to a middle band of difficulty:

| starting DockQ | n | surfaces, high/low gain | dominance, high/low gain |
|---|---|---|---|
| 0.00–0.15 | 39 | 2.0 / 2.0 | 58% / 66% |
| **0.15–0.35** | 43 | **2.0 / 1.0** | **41% / 93%** |
| 0.35–0.60 | 25 | 1.5 / 1.0 | 90% / 86% |
| 0.60–1.01 | 54 | 1.0 / 1.0 | 91% / 99% |

Among interfaces that started equally badly at 0.15–0.35, those that gained were split across two
surfaces at 41% dominance; those that did not were committed to one surface at 93%. **Being
undecided is what makes extra seeds valuable, and this is independent of difficulty.**

For several of the largest gainers the correct surface never forms even a 5% cluster while 2–9% of
poses still reach it — the correct answer lives in the tail of the distribution, which explains why
scaling beyond 50 seeds continues to pay on some targets.

---

## 9. Failure taxonomy and its consequence

Across 186 interfaces using the best data available per interface:

| status | n | share |
|---|---|---|
| solved (selected pose ≥ 0.23) | 122 | 65.6% |
| **confident decoy** — a good pose exists, ipTM ranks a wrong one above it | 37 | 19.9% |
| **no acceptable pose** — none generated at any depth examined | 27 | 14.5% |

These need opposite interventions. The 37 confident decoys cannot be fixed by more sampling — the
answer is already in the pool and the selector misses it. The 27 coverage failures cannot be fixed
by a better selector. Pooling them obscures both.

---

## 10. Caveats

- fnat is a component of DockQ; §4 is an attribution among components, not a causal test. The
  independent evidence that contact choice (not geometry) is what varies comes from §5, §6 and §7,
  which do not use fnat.
- Epitope clustering thresholds (Jaccard ≥ 0.5, ≥5% of poses) are choices. Single linkage biases
  toward *fewer* clusters, so the "commits to one surface" finding is conservative; but the 5% floor
  discards rare surfaces, and §8 shows the correct surface sometimes lives exactly there.
- 17 interfaces were excluded from the epitope-RMSD analysis for having fewer than 4 ground-truth
  epitope residues; OST's `nnat` confirms several of these are genuinely marginal interfaces
  (1–12 native contacts).
- Only 25 of 104 complexes have been scored on every pose on disk; the rest are scored on 250 poses.
  When the remaining poses of 40 interfaces were scored, 8 changed status, so some labels in §9 are
  provisional.
- Everything is one model (Protenix) on one benchmark. The CDR H3 relationship replicates the
  direction reported for AlphaFold3 by Hitawala & Gray, but the two papers may not use the same
  RMSD variant — their text does not specify which, so the numerical comparison is unresolved.

---

## 11. What this implies for method design

1. **Loop and fold modelling is a dead end for this problem.** Both sides of the interface are
   already built correctly in nearly every seed.
2. **The target is epitope selection**, and the decision is a discrete, near-tie choice among ~2
   candidate surfaces made in the trunk during recycling. Interventions should act there.
3. **Two distinct problems.** Roughly 20% of interfaces need a better *selector*; roughly 15% need
   better *coverage*. A method should declare which it addresses.
4. **The evidence is thin by construction.** For antibody–antigen the model has little real signal
   about where to dock: the paratope is nearly antigen-independent, templates carry no cross-chain
   geometry under the multichain mask, and the paired MSA is largely phylogenetic artefact because
   antibodies do not co-evolve with their antigens. A near-degenerate decision under weak evidence is
   exactly the regime in which stochastic perturbation flips the outcome — which is what seed scaling
   exploits.
