# Confident-Decoy Rescue — MASTER INDEX (point a new session HERE)

**This is the single entry point.** Everything below is either summarized here or linked with an
exact path. Read this file first; open the linked files only for detail. Last updated 2026-08-01.

---

## 1. The problem
Antibody–antigen interfaces where Protenix's ipTM confidently selects a structurally WRONG pose
(DockQ < 0.23) over 50 seeds, even though a correct pose exists in the 250-pose pool (seeds 101–150
× 5 samples). **15 decoy interfaces (12 unique pdbs).** Goal: a *deployable* rescue method.
Selection rule everywhere: top interface-pair_iptm pose's DockQ ≥ 0.23.

## 2. Headline results (self-contained)

| run | template | decoys /15 | goods /15 |
|---|---|---|---|
| free | none | **0** † | 15 (control) |
| fullmsa | none, but **no MSA row subsampling** | **3** † (net +2 vs free) | — |
| abonly | native **antibody only** | **8** | **15 (SAFE)** |
| agonly | native **antigen only** | 6 | 15 |
| oracle | best-**lDDT-to-native** fold, **per chain** (mix-match) | **8** | *(failed—bug)* |
| native50 | both chains **exact native** | 14 | 15 |
| Exp D | best-**pLDDT** free fold, per chain | 0 | — |
| sposefv | oracle w/ Fab H+L from **same pose** | 2/8 scored | — |
| topiptm | folds from **top-ipTM (decoy) pose** | 0/5 scored | — |
| topdockq | folds from **top-DockQ (best-docked) pose** | 1/14 scored | — |
| pddag | best-**pLDDT antigen** fold only (antibody freed) | **1** (8ux6_DF) | — |

> **† SEED-RANGE CAVEAT (2026-07-31).** "free = 0/15" is measured on the canonical CSV, which mixes
> seed ranges (0–49 for 8fbw/8q3j/8ux6/8yxi, 101–150 for the rest). **Seed-matched at 101–150, free is
> 1/15** (8ux6_DF rescued). The fullmsa row is 3/15 at 101–150, so the like-for-like delta is **+2, not
> +3**. See §2e — always state the seed range when comparing to free.

> `abonly` / `agonly` / `pddag` are the **label-corrected** numbers (a stale antigen-index file had
> mislabeled a heavy chain as the antigen for 8yxi + 8fbw; fixed and re-run 2026-08-01). Details in §2f.

### 2a. Isolation vs complex folding (per-chain fold quality — dead avenue)
Folded each chain at 50 seeds × 5 = 250 poses: **antigen alone** and **antibody as Fab H+L**
(nanobody alone), scored best-of-250 per-chain lDDT vs native, isolation vs the free-complex folds.
**Folding antibody+antigen TOGETHER is at-least-as-good everywhere, and clearly better for the
antigen** — isolation folding gives NO better template.

| chain | median Δ (iso best − cplx best) | iso better | verdict |
|---|---|---|---|
| antibody | −0.001 | 9/21 chains | dead even (coin-flip) |
| antigen | −0.015 | 2/12 chains | **complex better** |

Antigen gain is driven by peptide antigens with no stable isolated fold (need the Ab scaffold):
8fbw 17-mer iso 0.32→cplx **0.74**, 8fas 20-mer 0.71→**0.98**, 8che 0.59→0.65, 8ux6 0.86→0.91.
Whole-chain pLDDT-selection regret is small in median (antigen 0.021) but has a **fat tail (max
0.215 = 8che antigen)** — the exact hard cases a better antigen-fold selector would need to fix.
Data: `confident_decoy_experiments/artifacts/isolation_fold/isolation_vs_complex_fold_lddt.txt`;
scorer `scripts/fold_cmp2.py`. Memory: [[isolation-folds-not-better]].

### 2b. Antigen-fold localization — is the good antigen fold a structural OUTLIER? (2026-07-29)
Motivation: would leader/NMS compression surface the near-native antigen fold? Only if it is globally
*distinct* from the wrong folds. Scorer `scripts/localize_antigen.py`; data
`artifacts/antigen_localization/localization_results.json`. Key column = **lDDT(oracle fold vs
pLDDT-picked fold)** — what structural NMS actually "sees"; HIGH ⇒ same cluster ⇒ NOT separable.

| antigen | native res | oracle vs native | pLDDT-pick vs native | selection gap | **oracle vs pLDDT-pick** |
|---|---|---|---|---|---|
| 8the_q0 | 712 | 0.911 | 0.910 | 0.001 | 0.967 |
| 8slb_q0 | 248 | 0.886 | 0.882 | 0.004 | 0.923 |
| 8bf4_q1 | 491 | 0.902 | 0.895 | 0.007 | 0.947 |
| 8fbw_q0* | 231 | 0.937 | 0.929 | 0.008 | 0.979 |
| 8yxi_q0* | 218 | 0.898 | 0.885 | 0.013 | 0.953 |
| 8q3j_q0 | 145 | 0.849 | 0.836 | 0.013 | 0.957 |
| 8aok_q0 | 116 | 0.919 | 0.898 | 0.021 | 0.957 |
| 7z2m_q2 | 101 | 0.889 | 0.857 | 0.032 | 0.860 |
| 8av2_q0 | 191 | 0.884 | 0.824 | 0.060 | 0.899 |
| **8fas_q2** | **11 (pep)** | 0.984 | 0.604 | **0.380** | **0.517** |
| **8che_q2** | **15 (pep)** | 0.654 | 0.429 | **0.225** | **0.524** |
| **8fbw_q2** | **16 (pep)** | 0.537 | 0.349 | **0.188** | **0.478** |
| **8ux6_q2** | **20 (pep)** | 0.908 | 0.852 | 0.056 | **0.555** |
`*` 8fbw_q0/8yxi_q0 are HEAVY chains mis-scored as antigens (label bug, §2f); the real 8yxi antigen (q2,
347 res) was missed. Both rows sit with the other domains and don't change the conclusion.

**Conclusions:** (1) **Domain antigens: no selection headroom.** Oracle and pLDDT-picked folds are
0.90–0.98 lDDT *identical to each other* — one NMS cluster, so **leader-compression cannot separate
them** — and the selection gap is already ≤0.06 (mostly <0.02), i.e. pLDDT already picks near-oracle.
(2) **The antigen "selection problem" is almost entirely a PEPTIDE phenomenon** — peptides show gaps of
0.19–0.38 and are genuinely globally distinct (oracle-vs-pick ~0.5, so NMS *could* separate them), and
they alone drive the fat tail in §2a. (3) Epitope-localized differences were weak/mixed (2 cases
reversed), so epitope-region pLDDT would not cleanly separate them either.

### 2b-ii. pddag — deployable antigen template (2026-07-29, DONE: **1/15**)
Best-**pLDDT** antigen fold as the ONLY template, antibody freed = deployable analog of `agonly`
(native antigen → 8/15). Result **1/15** (only 8ux6_DF, DockQ 0.253, marginally over threshold);
independently re-tallied from `artifacts/pddag_scores.json`. Near-miss 8fas_BC 0.206. Confirms
prediction from §2b: a pLDDT-selected (non-exact) antigen template does NOT rescue. Note the
signature persists — high pair_iptm with low DockQ (8fas 0.90 ipTM → 0.21 DockQ, 8bf4 0.83 → 0.01):
a non-exact template RAISES ipTM without CRASHING the decoy's. **CONFIRMED 2026-08-01:** 8yxi_AC/8yxi_BC/
8fbw_AC originally used the wrong chain, and have now been **re-run with the correct antigen** (§2f) — all
three still fail (DockQ 0.072 / 0.148 / 0.173). **pddag stands at 1/15 with fully correct labels.**

### 2c. CDR-H3 in-pool analysis — selection vs generation (2026-07-29)
Does the model ever SAMPLE a native-matching CDR-H3? Per antibody heavy chain (identified by `kind()`
motif — immune to the label bug), scored H3-loop lDDT vs native over all 250 poses (seeds 101-150).
H3 = residues between the FR3 Cys and FR4 WGxG. Scorer `scripts/h3_inpool.py`; data
`artifacts/h3_inpool/h3_inpool_results.json`. **best-of-250 = does a native-like H3 EXIST in the pool.**

All values are H3-loop **lDDT vs native** (accuracy). "pick" columns = the H3 lDDT of the pose chosen by
that pLDDT criterion; Δ = H3-pLDDT-pick − global-pLDDT-pick (does a LOCAL H3 confidence select a better H3?).

| antibody | H3 len | best-of-250 | median | global-pLDDT-pick | H3-pLDDT-pick | Δ(H3−global) | regime |
|---|---|---|---|---|---|---|---|
| 8bf4_H2 | 5 | 0.950 | 0.915 | 0.926 | 0.935 | +0.009 | short: in-pool & picked |
| 8bf4_H0 | 7 | 0.925 | 0.852 | 0.913 | 0.774 | −0.139 | short: fine |
| 8che_H0 | 7 | 0.973 | 0.923 | 0.943 | 0.888 | −0.055 | short: fine |
| 8ux6_H1 | 10 | 0.855 | 0.804 | 0.800 | 0.800 | 0.000 | fine |
| **7z2m_H1** | 11 | **0.884** | 0.774 | **0.589** | 0.774 | **+0.185** | **SELECTION-limited** |
| 8fas_H0 | 13 | 0.990 | 0.946 | 0.983 | 0.978 | −0.005 | fine |
| 8av2_H1 | 14 | 0.726 | 0.525 | 0.491 | 0.548 | +0.057 | both (mediocre + missed) |
| 8yxi_H0 | 15 | 0.558 | 0.505 | 0.507 | 0.500 | −0.007 | GENERATION-limited |
| 8q3j_H2 | 19 | 0.683 | 0.564 | 0.595 | 0.505 | −0.090 | GENERATION-limited |
| 8aok_H1 | 20 | 0.759 | 0.608 | 0.608 | 0.598 | −0.010 | GENERATION-limited |
| 8the_H1 | 20 | 0.683 | 0.583 | 0.560 | 0.560 | 0.000 | GENERATION-limited |
| 8slb_H1 | 23 | 0.765 | 0.632 | 0.608 | 0.617 | +0.009 | GENERATION-limited |
| 8fbw_H0 | 23 | 0.965 | 0.859 | 0.922 | 0.922 | 0.000 | long but in-pool |

**H3-pLDDT vs global-pLDDT selection:** better in only 4/13, worse in 6/13, tied 3 — **mean Δ ≈ −0.003** (a
wash). Clear win only on 7z2m (+0.185, the one selection-limited case); actively hurts short easy H3s
(8bf4_H0 −0.139, 8che −0.055) where global pLDDT already nails it. So H3-region pLDDT is a *targeted*
fix for the selection-limited case, NOT a general selector.

**CAVEAT — lDDT is not a complete H3 metric.** lDDT is superposition-free and contact-density-weighted, so a
solvent-exposed loop tip (few neighbors) is under-weighted, and a rigid-body-shifted-but-locally-correct loop
can still score high. The binding-relevant metric is framework-aligned **CDR-H3 backbone RMSD** (Å).

**RMSD companion (framework-aligned CDR-H3 CA RMSD, Å) — scorer `scripts/h3_rmsd.py`, data
`artifacts/h3_inpool/h3_rmsd_results.json`. Framework superposition RMSD ~0.3–1.2 Å (alignment is sound).**
Convention: ≤1.5 Å near-native, 1.5–3 approximate, >3 wrong loop. best = MIN over 250 = closest H3 sampled.

| antibody | H3 len | best lDDT | **best RMSD Å** | global-pLDDT-pick Å | H3-pLDDT-pick Å | regime |
|---|---|---|---|---|---|---|
| 8bf4_H2 | 5 | 0.950 | 0.17 | 0.38 | 0.38 | native in pool |
| 8che_H0 | 7 | 0.973 | 0.20 | 0.40 | 0.73 | native in pool |
| 8fas_H0 | 13 | 0.990 | 0.22 | 0.27 | 0.43 | native in pool |
| 8bf4_H0 | 7 | 0.925 | 0.50 | 0.56 | 0.95 | native in pool |
| 8fbw_H0 | 23 | 0.965 | 0.51 | 1.03 | 1.03 | native in pool (long!) |
| 8ux6_H1 | 10 | 0.855 | 0.80 | 1.51 | 1.53 | near-native |
| **7z2m_H1** | 11 | 0.884 | 0.93 | 2.50 | 2.03 | **SELECTION-limited** |
| 8slb_H1 | 23 | 0.765 | 1.67 | **10.94** | 10.85 | one ok pose, pick catastrophic |
| 8av2_H1 | 14 | 0.726 | 1.94 | 5.80 | 5.80 | generation (+missed) |
| 8aok_H1 | 20 | 0.759 | 2.10 | 5.16 | 5.68 | generation |
| 8q3j_H2 | 19 | 0.683 | 3.04 | 4.57 | 6.52 | GENERATION FAIL |
| 8yxi_H0 | 15 | 0.558 | 5.00 | 8.51 | 5.61 | GENERATION FAIL |
| 8the_H1 | 20 | 0.683 | 5.02 | 9.65 | 9.49 | GENERATION FAIL |

**Conclusions (RMSD):**
- **The generation/selection split is ROBUST to the metric** — same 7 in-pool (best ≤~1 Å) vs 6 not (best
  1.7–5 Å) partition as lDDT. Reassuring: native H3 exists in the pool for 7 (incl. one 23-mer, 8fbw 0.51 Å);
  generation-fails for 6 (all long/hard). 7z2m is the lone selection-limited case (0.93 Å in pool, picks miss).
- **lDDT UNDERSTATED severity (validates the caveat):** 8yxi/8the lDDT 0.56/0.68 sound "meh" but best RMSD is
  **5.0 Å = totally wrong loop, never sampled close**. 8slb's global-pLDDT pick is lDDT 0.61 but **10.9 Å RMSD**
  (loop flung to the wrong place). lDDT 0.6 ≠ "roughly right." Use RMSD (or the functional template test) to
  judge H3 generation failure; lDDT compresses catastrophic misplacement into middling-looking scores.
- Operational defs adopted: **"good" H3 ≤1.5 Å; generation failure = best-of-250 never < ~2 Å** (8q3j/8yxi/8the).
- Still TODO (decisive functional test): does best-of-pool H3 as a template actually rescue? RMSD/lDDT are proxies.

**Conclusions:**
1. **Short H3 (≤13): native H3 IS in the pool AND pLDDT already picks it** (best 0.93–0.99, pick close). Not the decoy driver.
2. **Long H3 (≥14): mostly GENERATION-limited** — best-of-250 is only ~0.56–0.77 (5 of 6). The model
   never draws a native-quality long H3, so NO selection measure can help — must sample H3 harder.
   (Exception: 8fbw_H0 len-23 sampled well at 0.965.)
3. **Only 7z2m is cleanly SELECTION-limited** (native H3 in pool 0.884, pLDDT picks 0.589). **H3-region
   pLDDT recovers it to 0.774** — a local H3 signal helps here but is flat/negative elsewhere (not general).
4. **This explains the antibody bottleneck:** abonly's rescues (8aok/8av2/8q3j/8slb) are EXACTLY the long,
   generation-limited H3s (len 20/14/19/23). Native antibody template rescues by SUPPLYING the H3
   conformation the model can't generate — not by selecting better. Ties to [[error-localizes-to-cdrh3]].

### 2d. HOLO/APO — are native templates helping by ACCURACY or by LEAKING bound-state geometry? (2026-07-30)
**The concern (user, 2026-07-30):** native chains are lifted from the *complex*, so they are in the
**holo** (bound) conformation. "Native template" therefore conflates (i) accuracy — reproducible by any
predictor — with (ii) holo-ness — the interface geometry we are trying to predict, i.e. answer leakage.
If (ii) dominates, native50's 14/15 was never reachable, predicted monomers are apo-like by construction
and can never close the gap, and the "deployable ceiling" framing is misleading. (`apo` = "without",
`holo` = "whole/bound"; from apo-/holo-enzyme.)

**Evidence line 1 — same antigen, two different complexes (experimental).** FoldBench cross-match of all
12 decoy antigens against its 116 structures found **exactly one** re-used antigen: **8aok's antigen ↔
7yds chain A (97.9% id)** — PD-L1 bound to a *nanobody* in 8aok and to a *Fab* in 7yds. Conformational
difference between the two bound states: **global CA RMSD 1.06 Å, median per-residue 0.48 Å, 3/114
residues >2 Å.** ⇒ for this rigid domain, which partner binds barely changes the fold. (Caveat: n=1, and
holo-vs-holo, not apo-vs-holo. Every other antigen is unique to its complex, so FoldBench cannot supply
more; the 4 antibody-framework "matches" at 0.90–0.98 are framework conservation, not the same antibody.)

**Evidence line 2 — binding-dependence index (reinterpretation of §2a, no new compute).** The
isolation-vs-complex fold experiment IS a direct binding-influence assay: complex-best − isolation-best
lDDT = how much the partner's presence changes the fold the model produces.

| antigen | type | iso best | complex best | **binding-dependence** |
|---|---|---|---|---|
| 8fbw | 17-mer peptide | 0.323 | 0.738 | **+0.415** |
| 8fas | 20-mer peptide | 0.711 | 0.984 | **+0.273** |
| 8che | peptide | 0.592 | 0.654 | +0.062 |
| 8ux6 | peptide | 0.862 | 0.908 | +0.046 |
| 8aok / 8bf4 / 8yxi / 7z2m / 8av2 / 8the | domain | — | — | +0.020 … +0.003 |
| 8q3j / 8slb | domain | — | — | −0.006 / −0.010 |

**Verdict — the concern SPLITS BY ANTIGEN TYPE:**
- **Domain antigens: holo-ness does NOT appear to be the driver.** Binding-dependence ≈ 0 (−0.01…+0.02),
  corroborated independently by the experimental 8aok/7yds comparison (0.48 Å). There is barely any
  apo/holo difference available to leak, so native templates win on **accuracy**. The competing
  explanation is testable here and the data does not support it.
- **Peptide antigens: the concern is FULLY JUSTIFIED and unfixable.** A free 17–20-mer has no ordered
  unbound conformation — an apo structure cannot exist even in principle. The native peptide template is
  therefore almost pure holo information. **5 of oracle's 8 rescues are peptides**, so a large share of the
  headline rescue signal is structurally unearnable by any apo predictor.
- **Status: hypothesis-level, not settled.** Line 2 measures the *model's* binding-induced change, not the
  physical one; line 1 is n=1 and holo-vs-holo. A true apo control needs external PDB entries (feasible for
  well-studied domain antigens e.g. PD-L1; impossible for the peptides).

**Evidence line 3 — full binding-dependence assay, ALL chains, GLOBAL + INTERFACE (2026-07-30).**
Scorer `scripts/binding_dependence.py`; data `artifacts/binding_dependence/binding_dependence_results.json`.
32 chains (20 antibody, 12 antigen). Seed-matched pools: isolation (≤50 seeds) vs free complex (seeds
101-150), best-of-pool lDDT vs native, copy-aware interfaces (§6.7). `binddep = complex_best − isolation_best`.

| | n | binddep GLOBAL (median) | binddep INTERFACE (median) | complex-best global | complex-best interface |
|---|---|---|---|---|---|
| ANTIGEN | 12 | +0.015 | +0.019 | 0.907 | 0.885 (gap −0.022) |
| ANTIBODY | 20 | +0.003 | +0.010 | 0.942 | 0.888 (gap **−0.054**) |

**Column definitions (spelled out):**
- *Folded alone (isolation)* — the chain predicted WITHOUT its binding partner; best of ≤50 seeds × 5 samples.
- *Folded in complex (free run)* — the same chain taken from the full-complex prediction; best of 50 seeds
  (101–150) × 5 = 250 poses. Both scored as **lDDT vs the native chain**; "best" = max over the pool.
- *Whole-chain lDDT* — all residues of the chain.
- *Interface-only lDDT* — **epitope** for antigens (antigen residues within 5 Å of any antibody chain in the
  NATIVE structure) / **paratope** for antibody chains (that chain's residues within 5 Å of the antigen).
- *Change* = (in complex) − (alone). **Positive ⇒ the partner's presence is required to get the fold right
  ⇒ binding-dependent (holo) conformation ⇒ that native template leaks bound-state geometry.**

#### ANTIGEN chains (12) — epitope vs whole-chain

| Antigen | Type | Interface residues | Whole-chain: alone | Whole-chain: in complex | Whole-chain change | Epitope: alone | Epitope: in complex | **Epitope change** |
|---|---|---|---|---|---|---|---|---|
| 8fas q2 | peptide (20-mer) | 11 | 0.711 | 0.984 | +0.273 | 0.654 | 0.983 | **+0.329** |
| 8fbw q2 | peptide (17-mer) | 13 | 0.323 | 0.537 | +0.214 | 0.340 | 0.531 | **+0.192** |
| 8che q2 | peptide (37) | 15 | 0.592 | 0.654 | +0.062 | 0.593 | 0.648 | **+0.055** |
| 8ux6 q2 | peptide (41) | 16 | 0.862 | 0.908 | +0.046 | 0.834 | 0.883 | **+0.050** |
| 8av2 q0 | domain (198) | 30 | 0.874 | 0.884 | +0.010 | 0.828 | 0.851 | +0.023 |
| 8aok q0 | domain (117) | 23 | 0.899 | 0.919 | +0.020 | 0.875 | 0.898 | +0.022 |
| 8bf4 q1 | domain (523) | 57 | 0.890 | 0.905 | +0.015 | 0.840 | 0.857 | +0.017 |
| 8the q0 | domain (777) | 33 | 0.909 | 0.912 | +0.003 | 0.874 | 0.887 | +0.013 |
| 8slb q0 | domain (288) | 40 | 0.898 | 0.888 | −0.010 | 0.916 | 0.926 | +0.010 |
| 8yxi q2 | domain (347) | 26 | 0.918 | 0.933 | +0.015 | 0.948 | 0.952 | +0.004 |
| 7z2m q2 | domain (122) | 33 | 0.941 | 0.941 | 0.000 | 0.945 | 0.940 | −0.005 |
| 8q3j q0 | domain (152) | 27 | 0.863 | 0.849 | −0.014 | 0.851 | 0.842 | −0.009 |
| **median — peptides (4)** | | | | | **+0.054** | | | **+0.053** |
| **median — domains (8)** | | | | | **+0.007** | | | **+0.014** |

#### ANTIBODY chains (20) — paratope vs whole-chain

| Antibody chain | Type | Interface residues | Whole-chain: alone | Whole-chain: in complex | Whole-chain change | Paratope: alone | Paratope: in complex | **Paratope change** |
|---|---|---|---|---|---|---|---|---|
| 8bf4 q0 | heavy, H3 = 7 | 27 | 0.856 | 0.922 | +0.066 | 0.705 | 0.862 | **+0.157** |
| 8slb q1 | heavy, H3 = 23 | 18 | 0.893 | 0.934 | +0.041 | 0.635 | 0.758 | **+0.123** |
| 8the q1 | heavy, H3 = 20 | 20 | 0.859 | 0.886 | +0.027 | 0.559 | 0.644 | **+0.085** |
| 8fas q0 | heavy, H3 = 13 | 16 | 0.960 | 0.971 | +0.011 | 0.937 | 0.991 | +0.054 |
| 8bf4 q2 | heavy, H3 = 5 | 19 | 0.940 | 0.941 | +0.001 | 0.876 | 0.912 | +0.037 |
| 8av2 q1 | heavy, H3 = 14 | 34 | 0.849 | 0.853 | +0.004 | 0.722 | 0.754 | +0.032 |
| 8aok q1 | heavy, H3 = 20 | 21 | 0.888 | 0.890 | +0.002 | 0.691 | 0.722 | +0.030 |
| 8q3j q1 | light | 12 | 0.946 | 0.948 | +0.002 | 0.860 | 0.887 | +0.026 |
| 7z2m q1 | heavy, H3 = 11 | 18 | 0.942 | 0.953 | +0.011 | 0.871 | 0.889 | +0.018 |
| 8fas q1 | light | 7 | 0.969 | 0.966 | −0.003 | 0.985 | 0.996 | +0.011 |
| 8the q2 | light | 4 | 0.922 | 0.925 | +0.003 | 0.809 | 0.818 | +0.009 |
| 8yxi q0 | heavy, H3 = 15 | 4 | 0.901 | 0.898 | −0.003 | 0.581 | 0.590 | +0.008 |
| 8che q0 | heavy, H3 = 7 | 18 | 0.954 | 0.956 | +0.002 | 0.972 | 0.977 | +0.005 |
| 8yxi q1 | light | 15 | 0.927 | 0.926 | −0.001 | 0.926 | 0.927 | +0.001 |
| 8ux6 q0 | light | 16 | 0.937 | 0.944 | +0.007 | 0.937 | 0.938 | 0.000 |
| 8che q1 | light | 13 | 0.939 | 0.938 | −0.001 | 0.936 | 0.935 | −0.001 |
| 8fbw q1 | light | 5 | 0.936 | 0.946 | +0.010 | 0.934 | 0.932 | −0.002 |
| 8ux6 q1 | heavy, H3 = 10 | 19 | 0.948 | 0.950 | +0.002 | 0.879 | 0.873 | −0.006 |
| 8slb q2 | light | 12 | 0.963 | 0.966 | +0.003 | 0.954 | 0.944 | −0.010 |
| 7z2m q0 | light | 12 | 0.951 | 0.948 | −0.003 | 0.805 | 0.788 | −0.017 |
| **median — heavy chains (11)** | | | | | **+0.004** | | | **+0.030** |
| **median — light chains (9)** | | | | | **+0.003** | | | **+0.001** |

**Key reads from the per-chain tables:**
- Antigen: **peptides +0.053 median epitope change vs domains +0.014** — binding-dependence is a peptide effect.
- Antibody: **heavy-chain paratope change +0.030 vs whole-chain +0.004 (≈8× larger at the interface)**;
  light chains show nothing (+0.001). It is specifically the **heavy-chain paratope** that is partly induced.
- Paratope accuracy is far below whole-chain accuracy even in-complex: 8the q1 = 0.886 whole vs **0.644**
  paratope; 8yxi q0 = 0.898 vs **0.590**. Global lDDT badly oversells the binding-relevant region.
- **Open oddity:** 8bf4 q0 is the largest paratope mover (+0.157) despite a SHORT H3 (7) — breaks the
  "long H3 ⇒ induced" pattern; not yet investigated.

**Three conclusions:**
1. **Confirms the antigen split at the granular level:** peptide epitopes are strongly binding-dependent
   (+0.19…+0.33), domain epitopes are ~zero (+0.02…−0.01). Holo-leakage is a peptide phenomenon.
2. **NEW — the antibody PARATOPE is binding-dependent even when the chain globally is not.** Interface
   binddep median (+0.010) exceeds global (+0.003), and three heavy chains move 2–4× more at the paratope
   than globally (8bf4 +0.157 vs +0.066; 8slb +0.123 vs +0.041; 8the +0.085 vs +0.027). So the paratope
   conformation is *partly induced by binding* ⇒ **native ANTIBODY templates also leak holo geometry**,
   localized to the paratope. This is a NEW leak channel the global numbers hid, and it is consistent with
   abonly rescuing exactly the long-H3 cases (§2c).
3. **Interface regions are systematically less accurate than the whole chain** — antibody global 0.942 vs
   paratope 0.888 (−0.054); antigen 0.907 vs epitope 0.885 (−0.022). **Global lDDT overstates how well the
   binding-relevant region is modelled**, reinforcing §2c: always report the regional metric.

*Note:* 8fbw q2 complex-best reads 0.537 here vs 0.738 in §2a — §2a used the stored chain map, this uses
the copy-aware reference (§6.7). Treat the copy-aware numbers as authoritative.

**⚠ Benchmark anomaly found while doing this (2026-07-30, UNRESOLVED):** 7z2m's native assembly holds TWO
Fab copies and TWO antigen copies. Real binding units are **A/B–K** (12+18 contact residues) and **E/F–J**
(8+16). FoldBench's labelled interface **`7z2m_AJ` (A–J) touches at only 3 residues — a crystal-packing
contact, not the biological interface.** If DockQ is scored against that pair, the 7z2m_AJ decoy
designation may be measuring the wrong interface. **Not yet verified** how OST's chain mapping resolves
this — check before trusting any 7z2m result.

**Rescue sets (CORRECTED 2026-08-01, see §2f):**
abonly (8) = {8aok_AB, 8av2_AC, 8q3j_AC, 8slb_AB, 8slb_AC, **8yxi_AC**, **8yxi_BC**, **8fbw_AC**};
agonly (6) = {7z2m_AJ, 8bf4_AB, 8che_AE, 8che_BE, 8fbw_AC, 8ux6_DF};
oracle (8) = {8bf4_AB, 8che_AE, 8che_BE, 8fas_BC, 8fbw_AC, 8q3j_AC, 8ux6_DF, 8yxi_BC}.
**abonly ∩ agonly = {8fbw_AC}** (no longer empty). Union = 13/15; the 2 not rescued by either single chain
are **8the_AB** (needs both — native50 gets it) and **8fas_BC** (not rescued even by native50).
Partition: antibody-only 7, antigen-only 5, either-alone-works 1, needs-both 1, unrescuable 1.

### 2e. FULL MSA — does removing MSA random subsampling rescue the decoys? (2026-07-31, DONE: **3/15**, net **+2**)
**Question.** Every "free" run subsamples MSA rows at random per seed (§6.9). Does giving the trunk the
**full** MSA (`--data.msa.min_size.test 16384`, so `sample_size = n` always) fix the confident decoys?

**Setup.** All 12 PDBs / 15 interfaces, **seeds 101–150 × 5 samples = 250 poses each**, identical to the
original run in every way except MSA depth. Compared **seed-matched at 101–150 on both sides**.
Generation `scripts/fullmsa_queue_sched.py` (GPU work-queue, no idle GPUs); scoring
`scripts/fullmsa_n50_incr.py`; table `scripts/fullmsa_seedmatch_table.py` →
`artifacts/fullmsa_seedmatch_table.json`.

> **SEED-RANGE TRAP (cost one bogus analysis — read before comparing anything to the canonical CSV).**
> The canonical free CSV holds seeds **0–49** for 8fbw/8q3j/8ux6/8yxi and **101–150** for the other 8.
> Filtering it to 101–150 silently DROPS those 4 PDBs. Their 101–150 predictions existed on disk but were
> never scored; `scripts/score_free_101_150.py` scores them → `artifacts/free_101_150_scores.json`.
> **Consequence: the headline "free = 0/15" is seed-range dependent. At seeds 101–150 the original run is
> 1/15** — 8ux6_DF is rescued there (sel DockQ 0.254) but not at seeds 0–49 (0.225). Any free-vs-X claim
> must state its seed range.

**Column definitions** (per interface, over its own 250 poses; DockQ ≥ 0.23 = "acceptable" pose):
- `n_ok` — how many of the 250 poses are acceptable. **Coverage.**
- `top_ipt` — pair_iptm of the #1-ranked pose (what the model would actually hand you).
- `sel_dq` — DockQ **of that #1 pose**. `sel_dq ≥ 0.23` ⇔ **rescued**. This is the outcome column.
- `best_dq` — best DockQ anywhere in the pool = ceiling under perfect selection.
- `gap` = `top_ipt − pair_iptm(best acceptable pose)`. How much extra confidence the decoy holds over the
  best correct pose. **Smaller = ranking closer to correct; `+0.000` means rescued (true by construction,
  not independent evidence); `n/a` would mean no acceptable pose exists** (does not occur here — all 15
  have coverage in both runs).
- `dgap` = `gap(full MSA) − gap(original)`. **Negative = full MSA improved the ranking.**

| interface | n_ok | top_ipt | sel_dq | best_dq | gap | ‖ | n_ok | top_ipt | sel_dq | best_dq | gap | dgap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | *ORIGINAL (MSA-subsampled)* | | | | ‖ | | *FULL MSA* | | | | |
| 7z2m_AJ | 47 | 0.337 | 0.039 | 0.608 | +0.055 | ‖ | 37 | 0.276 | **0.352** ✅ | 0.562 | +0.000 | −0.055 |
| 8aok_AB | 109 | 0.749 | 0.112 | 0.695 | +0.025 | ‖ | 168 | 0.735 | **0.666** ✅ | 0.701 | +0.000 | −0.025 |
| 8che_BE | 24 | 0.443 | 0.087 | 0.529 | +0.034 | ‖ | 39 | 0.460 | **0.458** ✅ | 0.513 | +0.000 | −0.034 |
| 8ux6_DF | 10 | 0.377 | **0.254** ✅ | 0.320 | +0.000 | ‖ | 2 | 0.370 | 0.151 ❌ | 0.285 | +0.004 | +0.004 |
| 8av2_AC | 1 | 0.659 | 0.092 | 0.612 | +0.517 | ‖ | 4 | 0.629 | 0.092 | 0.745 | +0.231 | −0.286 |
| 8slb_AB | 65 | 0.601 | 0.012 | 0.462 | +0.297 | ‖ | 86 | 0.386 | 0.012 | 0.484 | +0.138 | −0.159 |
| 8slb_AC | 41 | 0.584 | 0.010 | 0.655 | +0.296 | ‖ | 41 | 0.375 | 0.010 | 0.516 | +0.143 | −0.153 |
| 8che_AE | 25 | 0.450 | 0.125 | 0.438 | +0.041 | ‖ | 41 | 0.444 | 0.090 | 0.477 | +0.003 | −0.038 |
| 8fbw_AC | 11 | 0.632 | 0.170 | 0.434 | +0.087 | ‖ | 11 | 0.620 | 0.120 | 0.244 | +0.099 | +0.012 |
| 8q3j_AC | 19 | 0.763 | 0.036 | 0.531 | +0.524 | ‖ | 15 | 0.795 | 0.044 | 0.754 | +0.553 | +0.029 |
| 8fas_BC | 77 | 0.857 | 0.175 | 0.987 | +0.092 | ‖ | 55 | 0.867 | 0.166 | 0.987 | +0.124 | +0.032 |
| 8yxi_AC | 7 | 0.293 | 0.006 | 0.295 | +0.093 | ‖ | 1 | 0.276 | 0.015 | 0.255 | +0.126 | +0.033 |
| 8yxi_BC | 4 | 0.281 | 0.009 | 0.538 | +0.088 | ‖ | 2 | 0.283 | 0.033 | 0.488 | +0.125 | +0.037 |
| 8the_AB | 62 | 0.608 | 0.113 | 0.514 | +0.336 | ‖ | 40 | 0.638 | 0.130 | 0.486 | +0.397 | +0.061 |
| 8bf4_AB | 22 | 0.857 | 0.014 | 0.723 | +0.012 | ‖ | 40 | 0.816 | 0.014 | 0.702 | +0.153 | +0.141 |

**Result: ORIGINAL 1/15 → FULL MSA 3/15 (net +2).** Gains {7z2m_AJ, 8aok_AB, 8che_BE}; **loss
{8ux6_DF}** (0.254 → 0.151). Full MSA both gains and loses cases.

**Interpretation — treat +2/15 as WEAK evidence, not a demonstrated effect:**
1. **No systematic ranking improvement.** Median `dgap` = **+0.004**; the gap shrinks in only **7/15**.
   That is a coin flip, which is what reshuffling (not recalibration) looks like.
2. **The mechanism is SELECTION, not generation or ipTM recalibration.** Fold quality is unchanged
   (median Δ per-chain lDDT = **0.000**), and there is no global ipTM shift. Coverage moves **both ways**
   (8aok 109→168, 8slb_AB 65→86 up; 8ux6 10→2, 8yxi_AC 7→1, 8fas 77→55 down).
3. **8bf4_AB is the cautionary case:** coverage nearly doubled (22→40) while ranking got *worse*
   (gap +0.012 → +0.153). It was a near-miss in the original run — a correct pose sat 0.012 ipTM under the
   decoy — and full MSA pushed it 0.153 away. **More correct poses ≠ better selection.**
4. Consistent with the project through-line: MSA subsampling is not the cause of the confident decoys.

### 2f. LABEL-BUG RE-RUN — corrected antigen labels for 8yxi + 8fbw (2026-08-01, RESOLVED)
**What was wrong.** `agonly_antigen_idx.json` named an antibody **heavy** chain as the antigen for 8yxi
(pointed at q0 heavy, missed the real antigen q2 = 347-res domain) and 8fbw (labeled q0 heavy + q2 peptide).
Runs that select a chain **by role** therefore templated the wrong chain on those 2 of 12 PDBs.

**Re-run.** Corrected labels → rebuilt templates → 8yxi + 8fbw re-run for all 3 role-based runs,
**seeds 101–150 × 5 = 250 poses per target, 9/9 targets scored**. Selection rule unchanged (top
interface-pair_iptm pose; rescued iff its DockQ ≥ 0.23).
Jobs `artifacts/relabel_jobs.json`; scores `artifacts/relabel_scores.json` + `relabel_scores_summary.json`.

Columns: `sel ipTM` = pair_iptm of the #1-ranked pose; `sel DockQ` = that pose's DockQ (**≥ 0.23 = rescued**);
`corrected` = verdict with the right chain templated; `was (buggy)` = what the old wrong-chain run reported.

| run | interface | sel ipTM | sel DockQ | corrected | was (buggy) |
|---|---|---|---|---|---|
| agonly (native antigen only) | 8yxi_AC | 0.335 | 0.015 | **not rescued** | RESCUED |
| agonly | 8yxi_BC | 0.306 | 0.029 | **not rescued** | RESCUED |
| agonly | 8fbw_AC | 0.873 | 0.764 | rescued | RESCUED (unchanged) |
| abonly (native antibody only) | 8yxi_AC | 0.864 | 0.789 | **RESCUED (new)** | no |
| abonly | 8yxi_BC | 0.889 | 0.939 | **RESCUED (new)** | no |
| abonly | 8fbw_AC | 0.763 | 0.586 | **RESCUED (new)** | no |
| pddag (best-pLDDT antigen only) | 8yxi_AC | 0.341 | 0.148 | not rescued | no (unchanged) |
| pddag | 8yxi_BC | 0.372 | 0.072 | not rescued | no (unchanged) |
| pddag | 8fbw_AC | 0.707 | 0.173 | not rescued | no (unchanged) |

**Consequences — this is not a bookkeeping fix, it reverses a headline claim:**
1. **agonly 8/15 → 6/15; abonly 5/15 → 8/15.** The 8yxi rescues were **antibody**-driven all along: they
   moved from the antigen column to the antibody column. 8fbw_AC's agonly rescue was real (it did template
   the true antigen among its labels), and 8fbw_AC is *additionally* rescued by abonly.
2. **"abonly ∩ agonly = ∅" IS NOW FALSE.** `8fbw_AC` is rescued by **both** (agonly 0.764, abonly 0.586).
   The clean either/or decomposition no longer holds.
3. **The direction of the bottleneck flips: antibody (8) now exceeds antigen (6), not the reverse.** Any
   argument that leaned on "antigen×8 / antibody×5" to call the antigen side dominant is inverted. Note this
   sits awkwardly beside Conclusion 2 (oracle says the *antigen* side is selection-limited) — the two are
   measuring different things (which chain, when given natively, suffices) and the tension is **unresolved**.
4. **pddag unchanged at 1/15** — with the *correct* antigen, best-pLDDT antigen templates still fail on both
   8yxi interfaces and 8fbw_AC (DockQ 0.072–0.173). Strengthens §2b-ii rather than revising it.

## 3. Conclusions
1. **Most decoys are bottlenecked on ONE chain — and it is more often the ANTIBODY** (antibody-only 7,
   antigen-only 5, either-alone-works 1 = 8fbw_AC, needs-both 1 = 8the, unrescuable-by-native 1 = 8fas).
   abonly is SAFE (goods 15/15). **REVISED 2026-08-01 (§2f):** pre-fix this read "antigen×8 / antibody×5
   disjoint"; correcting the 8yxi/8fbw antigen labels moved both 8yxi rescues from the antigen column to the
   antibody column (agonly 8→6, abonly 5→8) and broke disjointness (8fbw_AC is rescued by both).
2. **oracle 8/15 brackets Exp D (0) and native (14):** antigen side = **selection**-limited (good fold
   in pool, pLDDT blind, rank-corr ≈ +0.01); antibody side = **generation**-limited (best paratope
   fold only ~0.86 lDDT, oracle gets **3/8** of the antibody-bottlenecked cases — {8q3j_AC, 8yxi_BC,
   8fbw_AC}; was "1/5" before the §2f label fix). **TENSION (unresolved, 2026-08-01):** §2f now puts *more*
   decoys in the antibody column (8) than the antigen column (6), while this conclusion calls the antigen
   side the selection-limited one. The two measure different things — "which single native chain suffices"
   vs "can a pool fold be selected" — but the story is no longer clean. Worth re-deriving.
3. **pLDDT never picks the oracle fold (0/34 chains).** OK for antibody (tight pool 0.055), fatal for
   antigen (wide pool 0.121, good fold a rare outlier). Consensus (+0.19) also fails.
4. **ipTM is gated by interface FOLD QUALITY:** a good fold RAISES the good pose's ipTM; only an
   EXACT fold also CRASHES the decoy's ipTM. Deployable folds do the "raise good" half only.
5. **All "single pose's fold" ideas FAIL — mix-match oracle is the ceiling:** topiptm (decoy fold)
   reinforces the decoy; topdockq (best-dock fold) fails (best-DOCK ≠ best-FOLD, fold/dock decoupled);
   sposefv (same-pose Fv) ≈ oracle, not better.
6. **MSA random subsampling is NOT the cause of the confident decoys** (§2e). Full MSA moves 1/15→3/15
   seed-matched, but gains 3 and loses 1, leaves fold quality unchanged (median Δ lDDT 0.000), and does
   **not** systematically tighten the confidence gap (median dgap +0.004, better in 7/15). Coverage and
   ranking are decoupled — 8bf4 nearly doubled its acceptable poses while its ranking got worse.
7. **Through-line: "confident decoys all the way down."** Generation gives the right dock AND
   near-native folds; the confidence heads (ipTM for pose, pLDDT for fold) can't select them.

## 4. Open threads / next steps
- **The one open deployable lever: a better-than-pLDDT ANTIGEN-fold selector** (rare-outlier problem).
  Untested cheap idea: **epitope-region pLDDT** (interface-residue confidence, not whole-chain).
- topdockq 15th interface (7z2m_AJ) unscored (won't change ~1/15). Templates saved to re-run.
- **BUG:** oracle_goods fails — `prep_oracle_goods.py` can't write good-set native chain "AA0A" to
  legacy PDB (fix: mmCIF / remap chain IDs). abonly_goods 15/15 already covers safety.

## 4b. IDEAS TO TRY (parked — not scheduled, not tested)

### Unmask cross-chain template features (parked 2026-08-09)
**What.** Protenix zeroes every cross-chain template feature: `pairformer.py:1013` builds
`multichain_mask = (asym_id[:,None] == asym_id[None,:])` and `single_template_forward`
(`pairformer.py:1042-1085`) multiplies `template_distogram`, `template_unit_vector`,
`template_pseudo_beta_mask` and `template_backbone_frame_mask` by it. So a template transmits
intra-chain geometry ONLY — never a binding arrangement. Inherited from AF3, where templates are
retrieved per-chain by sequence search and the relative arrangement of two independently-found
templates is meaningless. Unmasking is a one-line change (drop the mask from those four products).

**Why NOT to do it inside the confident-decoy project:**
- It removes the property that makes the current result interpretable. Today, "templating on a pose
  improves docking" provably cannot be pose-copying; the donor's binding mode is invisible to the
  model. Unmasked, templating on a decoy would hand the model the decoy's binding mode as a prior —
  reinforcing exactly the pathology being fixed.
- Off-distribution. The model was TRAINED with the mask on, so those feature positions have only
  ever carried zeros. Whatever the pairformer does with non-zero values there is untested
  behaviour, not a capability.
- It breaks `native50` as an upper bound. 18/26 currently means "exact native FOLDS are worth
  18/26". Unmasked it would mean "native folds + the native docking pose", which is close to
  supplying the answer.

**Where it might be a real idea:** as separate work, it turns the model into a docking refiner —
feed an approximate complex, let it polish. Would need fine-tuning with the mask off for the
features to mean anything.

**Cheap sanity check if we ever want it:** unmask, template on the NATIVE complex, see whether it
beats 18/26. If it does not help even there, the channel is dead without retraining — one arm to
find out.

## 5. DEPENDENCY MAP (every path)

### Durable outputs (this project, under /home/ebrahim/)
- `confident_decoy_README.md` — THIS file (master index).
- `confident_decoy_FINDINGS.md` — full interpretation (+ "Session addendum 2026-07-28").
- `confident_decoy_RUNS.md` — per-run provenance log.
- `confident_decoy_experiments/RESULTS.md` — detailed results table.
- `confident_decoy_experiments/artifacts/scores/*.json` — RAW per-pose scores for every run
  (abonly, oracle, sposefv, topiptm, topdockq) + `reconstructed_summary.json`.
- `confident_decoy_experiments/configs/*.yaml` — reproducible per-run configs (motivation + pipeline).
- `confident_decoy_experiments/scripts/*.py` — all builders/schedulers/scorers/analysis (27 scripts).
- `confident_decoy_experiments/run_experiment.py` — replays a config's pipeline.
- `confident_decoy_experiments/artifacts/{meta.json,safety15.json,*_jobs.json,*_picks.json,*antigen_idx.json}`
  — interface defs + run inputs.

### Ephemeral working dir (session-specific /tmp — MAY BE WIPED, don't rely on it)
- `/tmp/claude-1003/-home-ebrahim/<session>/scratchpad/selftmpl/` — live outputs (`*_out/`),
  OST caches (`ost_*`), templates (`*_tmpl/`). Copies of the essential bits are in the durable
  locations above.

### Source data (inputs — stable)
- Interface defs: `confident_decoy_experiments/artifacts/meta.json` (15 decoys: pdb,c1,c2,chain_map),
  `safety15.json` (15 goods).
- Native structures: `/home/ebrahim/FoldBench/examples/ground_truths/{pdb}-assembly1.cif`.
- Free predictions (250-pose pool): `/home/ebrahim/Protenix/foldbench_abag/predictions_template/{pdb}/seed_*/predictions/`
  (NOTE: 4 regen pdbs 8fbw/8yxi/8q3j/8ux6 have their canonical free DockQ at seeds 0–49, others 101–150).
- Base input JSONs: `/home/ebrahim/Protenix/foldbench_abag/jsons/{pdb}-final-updated.json`.
- Canonical per-pose DockQ+ipTM (free run): `/home/ebrahim/tts_abag/results/csv_files/dockq_ost_protenix_full.csv`.
- Canonical OST scorer (match its flags): `/home/ebrahim/tts_abag/eval/eval_ost_foldbench.py`.
- Protenix inference: `cd /home/ebrahim/Protenix && .venv/bin/python runner/inference.py ... --use_template true`.
- OST binary: `/home/ebrahim/miniconda3/envs/foldbench/bin/ost` (LD_LIBRARY_PATH=.../foldbench/lib).
- mmCIF templates registered in `/home/ebrahim/mmcif/` + `/home/ebrahim/common/release_date_cache.json`.

### Memory (auto-loaded each session)
- `deployable-selection-sweep.md` (these results), `run-tracking-system.md` (logging rule),
  `error-localizes-to-cdrh3`, `native-rescue-mechanism`, `top1-selection-frontier`,
  `esmfold2-fast-is-handicapped`, `shared-server-resource-safety`.

### Unrelated but co-resident (leave alone)
- ESM 1024-seed run: `/home/ebrahim/esm/foldbench_predictions/esmfold2fast_tts/` (PAUSED; watchdog
  cron off; PAE .npy deleted for 50/99 complexes to stay under 1TB quota — see esm memories).
- Other GPU users on this box: `shawn` (GPUs 6–7), `john` (GPU5) — never touch their processes.

---

## 6. METHODS — how to set up a run (recipe; don't re-derive)

### 6.1 Protenix inference command (one process = one PDB, N seeds)
```
cd /home/ebrahim/Protenix
CUDA_VISIBLE_DEVICES=<gpu> LAYERNORM_TYPE=openfold OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path <PDB.json> --dump_dir <OUT> --seeds 101,102,... \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 --use_template true
```
- Python: `/home/ebrahim/Protenix/.venv/bin/python` (has gemmi 0.6.7; the base conda python does NOT).
- Output lands at `<OUT>/<name>/seed_<S>/predictions/<name>_sample_{0..4}.cif` + `<name>_summary_confidence_sample_{0..4}.json`, where `<name>` = the JSON's `"name"` field.
- Canonical seeds = **101–150** (seed-matched across free/agonly/native50/oracle/pddag). N_sample 5 ⇒ 250 poses. (Regen PDBs 8fbw/8q3j/8ux6/8yxi also have 0–49; 8 base PDBs got 0–100 added 2026-07-29, see RUNS `expand_seeds`.)

### 6.2 What "free" / template runs mean (chain-level templatesPath convention)
- **`--use_template true` is always on.** A chain is templated iff its `proteinChain["templatesPath"]` points to an a3m; set it to **`None` (drop the key) ⇒ that chain has NO template.**
- **free** (the 250-pose baseline, 0/15) = **no templates on any chain** (control).
- **native50** (14/15) = both antibody+antigen chains templated with the EXACT native fold.
- **agonly** (8/15) = antigen chain native-fold template, **antibody `templatesPath=None`**.
- **abonly** (5/15, SAFE) = antibody chain native-fold template, antigen freed.
- **oracle** (8/15) = each chain templated with its best-lDDT-to-native predicted fold (mix-match across poses).
- **Exp D** (0/15) = both chains templated with best-**pLDDT** predicted fold.
- **pddag** (2026-07-29) = antigen chain best-**pLDDT** predicted fold, antibody freed (deployable analog of agonly).
- Base JSONs carry an hmmsearch `templatesPath` by default — override or drop it per the run above.

### 6.3 Injecting a predicted fold as a template (the a3m + mmCIF trick)
Builder pattern (see `scripts/build_oracle_tmpl.py`, `build_pddag.py`; converter `cifconv.py`):
1. Register the pose CIF as a fake-old mmCIF template: `tid` = 4-char id (e.g. `o940`); write `convert(cif, tid)` to `/home/ebrahim/mmcif/<tid>.cif`; add `{tid:{"release_date":"2000-01-01"}}` to `/home/ebrahim/common/release_date_cache.json` (old date ⇒ passes the release-date filter). **Use a fresh tid range to avoid collision** (max used ~o932; pddag used o940+).
2. Write an a3m: line 1 `>query\n<seq>`; line 2 `>{tid}_{model_chain}/1-{L} [subseq from] mol:protein length:{L}  <tag>\n<seq-with-2-muts>` (mutate residues at L//3 and 2L//3 so it's not flagged identical).
3. Set `proteinChain["templatesPath"]` = that a3m. Templates carry ZERO cross-chain info (multichain_mask), so mixing chains from different poses is safe.

### 6.4 Which chain is antigen vs antibody
- `artifacts/agonly_antigen_idx.json` = `{pdb: [antigen qidx]}` (qidx = 0-based proteinChain order in the base JSON).
- `artifacts/oracle_chainmap.json` = `{pdb: {qidx: native_letter}}` (maps model chain → native chain letter).
- Peptide antigens (native <30 res: 8che/8fbw_q2/8fas/8ux6) behave differently from domain antigens — see §2a and FINDINGS.

### 6.5 Scoring & selection rule (concrete recipe — reuse `scripts/pddag_incr_score.py`)
Per pose, per interface `(c1,c2)` (native chain letters from `meta.json`), compute **[pair_iptm, DockQ]**:
1. **DockQ** — one OST call per pose CIF vs native:
   ```
   ost compare-structures -m <pose.cif> -r <pdb>-assembly1.cif -o <out.json> \
       --fault-tolerant --min-pep-length 4 --dockq        # taskset -c 0-49; LD_LIBRARY_PATH=.../foldbench/lib
   ```
   In `out.json`, find the `dockq_interfaces` entry `t` with `{t[0],t[1]} == {c1,c2}` (positions 0,1 = REFERENCE/native chains); `DockQ = dockq[k]` at that index. **Save `(t[2],t[3])` = the MODEL chains OST mapped** — needed next.
   (These are the exact flags that produced the agonly/oracle/Exp D numbers, so results stay comparable. Canonical eval `tts_abag/eval/eval_ost_foldbench.py` additionally passes `--lddt --min-nuc-length 4`; DockQ is unaffected. Per-residue lDDT: add `--local-lddt` — dict keyed `chain.resnum.`)
2. **pair_iptm** — from the pose's `<name>_summary_confidence_sample_<k>.json` field `chain_pair_iptm` (a matrix): `pair_iptm = M[i][j]` where `i=ord(t[2])-ord('A')`, `j=ord(t[3])-ord('A')` (the OST-mapped MODEL chains from step 1, A=0,B=1,…).
- **Selection rule (everywhere):** among the pose pool, take the pose with the highest interface `pair_iptm`; **rescued (decoy) / preserved (good) iff that pose's DockQ ≥ 0.23.** Re-tally from the score JSON, never from memory.
- **Scores JSON format:** `{iface: [[pair_iptm, dockq], ... one per pose]}`. Summary: `{rescued: "N/15", rescued_ifaces: [...]}`.
- **Reusable scorers** (`scripts/`): `pddag_incr_score.py` = incremental (scores each interface the moment its PDB's 50 seeds finish, writes running summary); `incr_score_generic.py` / `score_generic.py` = generic templates (args: out_dir, set, suffix, scores_json). One PDB's `<name>` dir may host 2 interfaces (8che/8slb/8yxi) — score both.
- OST is multithreaded: `OST_WORKERS≈40`, ThreadPool ~12–16, taskset-pin; verify measured CPU within 10s.

### 6.7 Interface (epitope/paratope) definition + COPY-AWARE chain resolution — read before any interface metric
**Interface residues:** residues of a chain with any heavy atom within **5.0 Å** of any heavy atom of the
partner chain(s), computed on the NATIVE assembly (`gemmi.NeighborSearch`). *epitope* = antigen residues
near any antibody chain; *paratope* = antibody-chain residues near the antigen.

**TRAP — native assemblies contain MULTIPLE COPIES.** Do NOT trust `oracle_chainmap.json` / `meta.json`
chain_map for interface work: they can mix copies or point at a crystal contact.
- 7z2m: Fab copies **A+B** and **E+F**, antigens **J** and **K**. The stored map takes light→E (copy 2) but
  heavy→B (copy 1) ⇒ heavy chain gets **0 paratope residues**. Real units: A/B–K, E/F–J (§2d anomaly).
- **Fix (implemented in `scripts/binding_dependence.py::resolve_unit`):** from all native chains matching a
  model chain by sequence, pick the **antigen copy with the most antibody contacts**, then each antibody
  chain's copy that contacts THAT antigen most. Verified: all 12 PDBs resolve with non-zero, sensible
  paratope/epitope sizes.
- **TRAP — partially-ordered peptides:** the construct sequence can far exceed what is modelled (8che: 37-mer
  in the JSON, only **15** residues in the crystal). Strict identity matching drops them entirely — use a
  **containment** score (longest common block / min length) alongside the ratio.
- **TRAP — cache keys:** if a template/scoring cache key omits the native chain letter, changing the copy
  assignment silently reuses stale results. Include the letter in the key.

### 6.8 Per-residue / regional lDDT (for epitope, paratope, CDR-H3 metrics)
`ost compare-structures -m <model> -r <native> -o out.json --lddt --local-lddt --fault-tolerant
--min-pep-length 4` → `local_lddt` is a dict keyed **`"<chain>.<resnum>."`** with `None` for unaligned
residues. To average over a region defined on NATIVE numbering, align model→native sequence
(`difflib.SequenceMatcher(..., autojunk=False)` — **autojunk MUST be off** for sequences ≥200 res) and map
model resnum → native resnum through the matching blocks.
**CDR-H3 definition used:** residues between the FR3 Cys and the FR4 `WGxG` motif (`scripts/h3_inpool.py`).
**Framework-aligned H3 RMSD** (`scripts/h3_rmsd.py`): superpose on VH framework CAs (VH minus H3, CH1
excluded), then CA RMSD over H3 only; report framework RMSD too as an alignment sanity check.
**Interpretation:** lDDT is contact-density weighted and superposition-free, so it UNDERSTATES a
displaced-but-locally-formed loop (see §2c: lDDT 0.61 ↔ 10.9 Å). Use RMSD for loops.

### 6.9 MSA handling at inference — RANDOM ROW SUBSAMPLING (this is the per-seed randomness source)
**Every forward pass randomly subsamples the MSA.** `protenix/model/utils.py::sample_indices` (line ~306):
```python
sample_size = torch.randint(low=min(lower_bound, n), high=n + 1, size=(1,)).item()  # UNIFORM in [1, n]
indices     = torch.randperm(n=n, device=device)[:sample_size]                      # random rows
```
Called from `protenix/model/modules/pairformer.py:854`
(`sample_msa_feature_dict_random_without_replacement`) inside the **MSAModule**, i.e. inside the trunk,
on **both train and eval paths** (only `cutoff`/`lower_bound` differ by mode).

Effective inference settings (`configs/configs_data.py:243` "msa" block):
| setting | test value | meaning |
|---|---|---|
| `min_size.test` (`test_lowerb`) | **1** | lower bound of the uniform draw |
| `sample_cutoff.test` (`test_cutoff`) | **16384** | cap applied AFTER sampling |
| `max_size.test` | 16384 | cap on MSA loaded from disk |
| `strategy` | `"random"` | `random` = randperm; `topk` = first-k (deterministic) |
| `msa_max_size` (model) | 16384 | padding size in `MSAModule` |

**Consequences (important for interpreting any seed-to-seed analysis):**
- **The number of MSA rows used is drawn UNIFORMLY from [1, n] on every pass** — so one seed may see 1
  sequence and another all n. It is NOT MSA-depth truncation. Confirmed the earlier assumption "MSA
  handling is deterministic" was WRONG.
- ⚠️ **CORRECTED 2026-08-09 — this is NOT the dominant source of between-seed variation.** Turning MSA
  subsampling OFF entirely (`fullmsa` arm, `--data.msa.min_size.test 16384`) left seed-to-seed variance
  essentially unchanged: mean SD of per-seed best DockQ **0.1311 → 0.1289** across 15 interfaces,
  lower on only **8/15** (`scripts/msa_seed_variance.py`, `artifacts/msa_seed_variance.json`). It is also
  not the only TRUNK-level source — MC dropout fires on ~40% of seeds (see §6.11).
- MSA file loading itself IS deterministic truncation (`msa_featurizer.py` `[:us]`/`[:ps]`, top-N), and our
  deepest MSA is ~3,260 sequences (8the) vs the 16384 cap, so **no MSA content is being dropped on load** —
  the randomness is purely the per-pass row draw.
- Within a seed, the trunk runs once, so all 5 diffusion samples share ONE MSA draw (see §6.10 decomposition).
- To make the trunk deterministic for an ablation, set `strategy: "topk"` (or raise `min_size.test` to n).

### 6.10 Bias/variance decomposition of a structural region (H3, epitope)
Trunk runs ONCE per seed then N_sample diffusion samples are drawn from the same `s,z`
(`protenix/model/protenix.py`: recycling at line 235, sampling at 530→572). So with coordinates in a
common frame (framework-superposed for H3; non-epitope-superposed for epitope):
- **BIAS** = RMSD(per-atom mean structure over all poses, native) — irreducible by sampling
- **DIFFUSION** = spread of poses about their own seed's mean (trunk fixed)
- **TRUNK** = spread of seed-means about the global mean (MSA draw + recycling vary)
- Identity: `mean_MSD_to_native = BIAS² + DIFFUSION² + TRUNK²` (verify it holds; it is exact).
**Do NOT use "SD of the RMSD values" as the variance term** — RMSD is a distance, so its spread does not
measure structural spread (poses scattered in different directions at equal distance give a tiny SD).
Scripts: `scripts/h3_variance_decomp.py`, `scripts/epitope_variance_decomp.py`, `scripts/proper_bias_variance.py`.
**Result (2026-07-30):** bias = **72% (H3) / 81% (epitope)** of total MSD; median DIFFUSION 0.59 Å vs
TRUNK 0.88 Å (H3). Errors are dominated by systematic bias, and diffusion is NOT the main culprit.

### 6.11 ALL FIVE SOURCES OF TEST-TIME RANDOMNESS (audited 2026-08-09 — read before any variance claim)
A seed sets the global torch RNG, so **all five vary together across seeds**. Two act through the TRUNK
(they change `z`, so all 5 diffusion samples of that seed inherit them); three act per DIFFUSION SAMPLE.

| # | source | code | scope |
|---|---|---|---|
| 1 | MSA row subsampling | `model/utils.py:306-308` (see §6.9) | trunk, redrawn per recycle (`N_cycle=4`) |
| 2 | **MC dropout on the recycle update** | `model/protenix.py:242` | **trunk, per seed** |
| 3 | diffusion initial noise | `model/generator.py:192` | per sample |
| 4 | churn noise (predictor-corrector) | `model/generator.py:212` | per sample, ×`N_step`(=200) |
| 5 | centre random augmentation | `generator.py:201` → `utils.py:28-94` | per sample, ×200 |

**#2 IS THE ONE THAT IS EASY TO MISS — and I did miss it.** `runner/inference.py:176` calls
`self.model.eval()`, which is why it *looks* like dropout is off. It is not:
```python
# protenix/model/protenix.py:405 (main_inference_loop), per model seed
mc_dropout = random.random() < mc_dropout_apply_rate        # configs_base.py:109 -> 0.4
# protenix/model/protenix.py:242, inside the recycling loop
if mc_dropout:
    z = z_init + F.dropout(self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)),
                           p=self.configs.mc_dropout_rate)  # configs_base.py:110 -> 0.4
```
`F.dropout`'s signature is `(input, p=0.5, training=True, ...)` — **`training` defaults to True and is not
passed here**, so the call applies dropout regardless of eval mode. `runner/inference.py:233` feeds the
config value in, so **every run in this project has had MC dropout active on ~40% of seeds at rate 0.4.**
(The `.eval()`/`.train()` pairs at `protenix.py:193-196, 299-303` are gated on `train_confidence_only` and
are irrelevant at inference.)

**Consequences for anything we have claimed about variance:**
- ~~"MSA subsampling is the dominant / only trunk-level source"~~ — **RETRACTED.** There are two trunk
  sources, and the `fullmsa` arm only removed one (see the correction in §6.9).
- ~~"diffusion variance dominates"~~ — **RETRACTED, was never supported.** It was inferred from
  fullmsa leaving between-seed variance unchanged, which is fully explained by #2 still firing.
  It also compared incommensurate statistics (mean range-of-5 vs SD of per-seed best).
- **To isolate the trunk properly** an ablation must set BOTH `--data.msa.min_size.test 16384` (or
  `strategy: topk`) AND `mc_dropout_apply_rate=0`. No arm has done this. The proper decomposition
  (§6.10) remains the right method; it just has not been run under those conditions.
- Anything that must be reproducible bit-for-bit needs `mc_dropout_apply_rate=0` pinned, not just a seed.

### 6.6 Multi-GPU scheduler + resource rules
- Pattern: `scripts/{expand_seeds,pddag}_sched.py` — one PDB per GPU, dynamic refill, detects in-flight `inference.py` by scanning `/proc/*/cmdline`+`environ` (so it won't double-book a GPU another job uses). **Create the log dir before `nohup` redirect** (the shell redirect runs before the script's `makedirs`).
- Default GPUs 0–3 for this project. **Never touch GPU 5 (john) or 6–7 (shawn).** Hard ceilings: 128 workers / 512 GB. **Check MEASURED usage (`nvidia-smi`, `ps`) within 10s of every launch** and kill if over.
