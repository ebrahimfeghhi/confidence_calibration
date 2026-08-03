# Confident-Decoy Report

Every result carries a `code:` line (the script that produced it) and a `data:` line (the artifact it
can be re-tallied from). A number without both is marked PROVISIONAL.
Paths relative to `/home/ebrahim/confident_decoy_experiments/`. Started 2026-08-01.

---

## 1. The benchmark: 15 confident decoys

A **confident decoy** is an antibody–antigen interface where the model's own confidence picks a wrong
pose even though a right one is in the pool it generated. Over the pose pool, both must hold:

1. **Selection fails** — the top interface-`pair_iptm` pose has **DockQ < 0.23**.
2. **A correct pose exists** — at least one pose has **DockQ ≥ 0.23**.

Condition 2 is what makes this a *selection* problem rather than a *docking* problem.

Preliminary analyses use **seeds 101–150 (250 poses per interface)**.

### Generation

Free run — MSAs only, no templates. One process per PDB:

```bash
cd /home/ebrahim/Protenix
CUDA_VISIBLE_DEVICES=<gpu> LAYERNORM_TYPE=openfold OMP_NUM_THREADS=4 \
.venv/bin/python runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path foldbench_abag/jsons/<pdb>-final-updated.json \
  --dump_dir foldbench_abag/predictions_template \
  --seeds 101,102,...,150 \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 --use_template false
```

Seeds differ because MSA rows are randomly subsampled at inference (row count drawn uniformly in
`[1, n_msa]`, then a random permutation). Details: `confident_decoy_README.md` §6.9.

### Scoring and selection

`scripts/decoy_set.py` scores every pose with OST and applies both conditions:

```bash
ost compare-structures -m <pred>.cif -r <native>.cif -o <out>.json \
    --fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq
```

`pair_iptm` is read from each pose's `_summary_confidence_sample_*.json` at the OST-mapped model chains.

### The set

`contacts` = residues within 5 Å across the native interface (sanity check that the interface is real).
`n_ok` = poses with DockQ ≥ 0.23. `top ipTM` = confidence of the #1-ranked pose. `sel DockQ` = that
pose's DockQ (decides the verdict). `best DockQ` = best pose in the pool — the ceiling perfect
selection would reach.

| interface | chains | contacts | n_ok | top ipTM | sel DockQ | best DockQ |
|---|---|---|---|---|---|---|
| 7z2m_AK † | A–K | 19 | 2 | 0.374 | 0.090 | 0.433 |
| 7z2m_BK † | B–K | 31 | 71 | 0.406 | 0.027 | 0.384 |
| 8aok_AB | A–B | 46 | 109 | 0.749 | 0.112 | 0.695 |
| 8av2_AC | A–C | 55 | 1 | 0.659 | 0.092 | 0.612 |
| 8bf4_AB | A–B | 51 | 22 | 0.857 | 0.014 | 0.723 |
| 8che_AE | A–E | 32 | 25 | 0.450 | 0.125 | 0.438 |
| 8che_BE | B–E | 24 | 24 | 0.443 | 0.087 | 0.529 |
| 8fas_BC | B–C | 11 | 77 | 0.857 | 0.175 | 0.987 |
| 8fbw_AC | A–C | 31 | 11 | 0.632 | 0.170 | 0.434 |
| 8q3j_AC | A–C | 34 | 19 | 0.763 | 0.036 | 0.531 |
| 8slb_AB | A–B | 46 | 65 | 0.601 | 0.012 | 0.462 |
| 8slb_AC | A–C | 20 | 41 | 0.584 | 0.010 | 0.655 |
| 8the_AB | A–B | 50 | 62 | 0.608 | 0.113 | 0.514 |
| 8yxi_AC | A–C | 13 | 7 | 0.293 | 0.006 | 0.295 |
| 8yxi_BC | B–C | 31 | 4 | 0.281 | 0.009 | 0.538 |

**15 interfaces, 12 unique PDBs** (7z2m, 8che, 8slb, 8yxi contribute 2 each).

These are not near-misses. On 8bf4, 8fas and 8q3j the model reports high interface confidence
(0.86, 0.86, 0.76) for poses scoring 0.014, 0.175 and 0.036, while poses at 0.72, 0.99 and 0.53 sit
unselected in the same pool. **Coverage is never the bottleneck — every interface has at least one
acceptable pose.**

> **† 7z2m relabelled.** FoldBench lists 7z2m's interface as **A–J**, but the assembly holds two copies
> of the same complex — Fab(A,B)+antigen **K**, and Fab(E,F)+antigen **J**. Chains A and J are from
> *different* copies and touch at only **3 residues** (a crystal contact, not binding). Likely cause: a
> representative chain picked per entity independently, first-alphabetically — light chains are {A,E}→A,
> antigens are {J,K}→J, which lands on two different copies. Corrected here to the real interfaces
> **A–K** and **B–K**, matching FoldBench's own one-row-per-(antibody chain, antigen) convention.
> 7z2m remains a confident decoy either way; the fix makes it measure the right interface.
>
> **Scored by the standard protocol, not a bespoke one.** An earlier version of this report subset the
> native to each interface pair and handed OST the full model ("copy-optimal mapping"). That is not the
> procedure in `tts_abag/eval/eval_ost_foldbench.py`, and it was wrong: scoring A–K and B–K in separate
> calls let OST choose a *different* predicted copy for each, so on 30 of 50 poses the two interfaces
> measured different physical Fabs. Both are now read from **one** OST call on the full native assembly
> vs the full model, under OST's own global chain mapping, extracted by name — so A–K and B–K always
> describe the same predicted Fab. This changed 7z2m's numbers (AK acceptable poses 11 → 2, BK 77 → 71);
> both remain confident decoys.
>
> **Audit of all 172 FoldBench ab–ag rows** (`scripts/foldbench_label_audit.py` →
> `artifacts/foldbench_label_audit.json`): **7z2m is the only clear instance.** 16 rows have <10 contact
> residues, but 14 are single-copy assemblies with genuinely small interfaces (5–9 residues), not
> mislabels. One other — **8udz A–C** (8 residues, multi-copy, same-entity alternative A–E at 18) — looks
> like the same cross-copy bug and is worth checking if 8udz is ever used. It is not in our set.
>
> `scripts/decoy_set.py` now **raises** if any scored interface has <10 contact residues, so a crystal
> contact cannot silently re-enter the benchmark.

### Provenance

- **code:** `scripts/decoy_set.py` — scores all poses via OST and applies both conditions. Reproduces the
  table above from raw structures; OST outputs cached in `artifacts/decoy_set_ost/`.
- **code:** `scripts/foldbench_label_audit.py` — the label audit described in the footnote.
- **data:** `artifacts/decoy_set_101_150.json` — per-interface `contact_residues`, `n_acceptable`,
  `top_pair_iptm`, `sel_dockq`, `best_dockq`, `confident_decoy`.
- **data:** `artifacts/foldbench_label_audit.json` — all 172 audited rows.
- **predictions:** `/home/ebrahim/Protenix/foldbench_abag/predictions_template/<pdb>/seed_<101-150>/predictions/`
- **ground truth:** `/home/ebrahim/FoldBench/examples/ground_truths/<pdb>-assembly1.cif`
- **interface definitions:** `artifacts/meta.json`

### Gotchas

- **Chain IDs: `label_asym_id`, not `auth_asym_id`.** FoldBench's targets CSV, `meta.json` and OST's
  `dockq_interfaces` all use `label_asym_id`; `gemmi.read_structure()` names chains by `auth_asym_id`.
  They differ for 8bf4, 8che, 8fbw, 8slb and 8the. Use `native_chains()` in `scripts/decoy_set.py`;
  hand-rolling this silently compares the wrong chains.
- **Seed range matters.** `8ux6_DF` is a decoy at seeds 0–49 (sel DockQ 0.225) but not at 101–150
  (0.254), so it is excluded here. Older tables that mix seed ranges will disagree. Always state the
  seed range.

---

## 2. Fold quality at the binding site: decoys vs goods

**Hypothesis (2026-08-01).** Where the model folds the paratope and epitope accurately, it has a good
structural representation of the binding site and does not invent a confident-but-wrong docking mode.
Where those regions are inaccurate, it mistakes a wrong site for a real one → high ipTM, low DockQ.

**Prediction.** Goods fold better than decoys, and the gap should be **larger at the paratope/epitope
than whole-chain**. If whole-chain separates them equally well, the story is only "goods are easier".

### Setup

- **Decoys:** the 15-interface set from §1. **Goods:** 15 control interfaces where selection already
  works (`artifacts/safety15.json`); verified **15/15 still good at seeds 101–150**, selected-pose
  DockQ 0.59–0.93.
- **Domain antigens only** (native antigen ≥ 30 residues). Peptides have no independently stable fold,
  so "fold accuracy" is ill-posed and confounded with the docking being tested. Excludes 4 decoys
  (8che ×2, 8fas, 8fbw) and 1 good (8gho). **Final n: 11 decoys, 14 goods.**
- **Poses.** Decoys are measured at *both* the top-ipTM pose (what the model selects — the wrong one)
  and the best-DockQ pose (the correct pose that exists in the pool). Goods at the top-ipTM pose only,
  since there it is already correct.
- **lDDT** vs the native chain, single-chain vs single-chain, so it is independent of where the pose
  docked. **pLDDT** from the model's B-factor column. Paratope/epitope = native residues within 5 Å of
  the partner chain.

### Result — the effect is localised to the PARATOPE

Median over interfaces. "gap" = good − decoy@top-ipTM; **larger = goods fold better.**

| metric | decoy @ top-ipTM | decoy @ best-DockQ | good @ top-ipTM | gap |
|---|---|---|---|---|
| antibody whole-chain lDDT | 0.915 | 0.922 | 0.939 | +0.024 |
| **paratope lDDT** | **0.614** | **0.760** | **0.926** | **+0.312** |
| antigen whole-chain lDDT | 0.884 | 0.896 | 0.896 | +0.012 |
| epitope lDDT | 0.864 | 0.898 | 0.906 | +0.042 |
| antibody whole-chain pLDDT | 0.869 | 0.854 | 0.911 | +0.042 |
| paratope pLDDT | 0.783 | 0.691 | 0.930 | +0.147 |
| antigen whole-chain pLDDT | 0.832 | 0.820 | 0.875 | +0.043 |
| epitope pLDDT | 0.857 | 0.773 | 0.929 | +0.073 |

**The hypothesis is supported, in a specific and narrow form:**

1. **Paratope accuracy separates the two sets; whole-chain accuracy does not.** The paratope gap
   (+0.312) is **13× the antibody whole-chain gap** (+0.024). Decoy antibodies are globally well folded
   (0.915) but their binding loops are not (0.614). This is exactly the localisation the hypothesis
   predicts, and it rules out "goods are simply easier targets".
2. **It is an ANTIBODY effect, not an antigen one.** Epitope gap is +0.042 and antigen whole-chain
   +0.012 — small either way. The binding-site inaccuracy that tracks false confidence sits on the
   paratope.
3. **Within decoys, the correctly-docked pose has a better paratope** (0.760 vs 0.614). Paratope
   accuracy and docking correctness move together pose-by-pose, not just target-by-target.
4. **The model is more confident about the paratope when it is wrong.** Paratope pLDDT is *higher* at
   the top-ipTM (wrong) pose than at the best-DockQ (right) pose, 0.783 vs 0.691. Confidence is
   anti-correlated with correctness exactly where the hypothesis says the error originates.

### Per-interface paratope lDDT

| decoy | whole-chain | paratope @ top-ipTM | paratope @ best-DockQ |
|---|---|---|---|
| 8q3j_AC | 0.876 | 0.525 | 0.539 |
| 8yxi_AC | 0.882 | 0.526 | 0.537 |
| 8the_AB | 0.854 | 0.528 | 0.526 |
| 8aok_AB | 0.843 | 0.565 | 0.702 |
| 8av2_AC | 0.801 | 0.608 | 0.752 |
| 8slb_AB | 0.915 | 0.614 | 0.769 |
| 7z2m_AK | 0.937 | 0.751 | 0.760 |
| 7z2m_BK | 0.931 | 0.818 | 0.831 |
| 8bf4_AB | 0.917 | 0.830 | 0.859 |
| 8yxi_BC | 0.911 | 0.911 | 0.902 |
| 8slb_AC | 0.962 | 0.929 | 0.932 |

| good | whole-chain | paratope @ top-ipTM |
|---|---|---|
| 9fww_AB | 0.836 | 0.516 |
| 8ido_AD | 0.863 | 0.763 |
| 8be3_AB | 0.917 | 0.870 |
| 8aom_AB | 0.937 | 0.884 |
| 8qf4_AB | 0.952 | 0.918 |
| 8c3l_AB | 0.909 | 0.920 |
| 8weo_AB | 0.941 | 0.925 |
| 8c5h_AB | 0.925 | 0.927 |
| 8hxq_AB | 0.913 | 0.934 |
| 8qz3_AC | 0.958 | 0.940 |
| 7xld_AB | 0.958 | 0.965 |
| 8pih_AB | 0.970 | 0.977 |
| 8q7o_AC | 0.966 | 0.978 |
| 8w90_AB | 0.960 | 0.979 |

**Overlap is real, so this is a tendency, not a clean separator.** 3 of 11 decoys have good paratopes
(8yxi_BC 0.911, 8slb_AC 0.929, 8bf4_AB 0.830) and one good has a poor one (9fww_AB 0.516). Medians differ sharply;
the distributions touch. n = 11 vs 14, no significance test run.

### Provenance

- **code:** `scripts/fold_quality.py` (decoys + goods, both poses, lDDT + pLDDT)
- **code:** `scripts/decoy_set.py --set goods` — goods DockQ at seeds 101–150
- **data:** `artifacts/fold_quality.json` — per-interface, per-pose, all 8 metrics
- **data:** `artifacts/goods_set_101_150.json` — goods selection verdicts

### Caveats

- Excluded 5 peptide-antigen interfaces; conclusions cover domain antigens only.
- **Two bugs were found and fixed in this analysis; the numbers above are post-fix.** (a) An earlier
  version handed OST the full model against a one-chain native, which mapped the wrong predicted chain
  and returned lDDT ≈ 0.004. (b) Pose selection read `dockq[0]` instead of matching the requested
  interface, so PDBs contributing two interfaces (7z2m, 8slb, 8yxi) silently scored the same one twice.
  Bug (b) affected a first published version of this table; the paratope gap was unchanged at +0.312,
  antigen/epitope numbers moved slightly, and the per-interface rows for those PDBs changed.
- Correlational. It does not establish that fixing the paratope fixes selection — though the native-template
  experiments (rescue 14/15 by collapsing the decoy's ipTM) point the same way.

---

## 3. Does good paratope folding mean a smaller confidence gap?

§2 says bad paratope folding tracks false confidence. If that is causal, decoys whose paratope **is**
well folded should be *closer* to ranking a correct pose first.

**Confidence gap** = `pair_iptm(top pose) − pair_iptm(most confident acceptable pose)`, where
acceptable = DockQ ≥ 0.23. A gap of 0 would mean the correct pose already ranks first.

| interface | paratope lDDT | epitope lDDT | top ipTM | top DockQ | best-acceptable ipTM | best-acceptable DockQ | confidence gap |
|---|---|---|---|---|---|---|---|
| 8q3j_AC | 0.525 | 0.840 | 0.763 | 0.036 | 0.239 | 0.525 | +0.524 |
| 8yxi_AC | 0.526 | 0.900 | 0.293 | 0.006 | 0.200 | 0.238 | +0.093 |
| 8the_AB | 0.528 | 0.879 | 0.608 | 0.113 | 0.273 | 0.233 | +0.336 |
| 8aok_AB | 0.565 | 0.864 | 0.749 | 0.112 | 0.724 | 0.681 | +0.025 |
| 8av2_AC | 0.608 | 0.796 | 0.659 | 0.092 | 0.142 | 0.612 | +0.517 |
| 8slb_AB | 0.614 | 0.843 | 0.601 | 0.012 | 0.304 | 0.349 | +0.297 |
| 7z2m_AK | 0.751 | 0.925 | 0.370 | 0.051 | 0.321 | 0.320 | +0.049 |
| 7z2m_BK | 0.818 | 0.935 | 0.400 | 0.034 | 0.389 | 0.261 | +0.011 |
| 8bf4_AB | 0.830 | 0.776 | 0.857 | 0.014 | 0.845 | 0.695 | +0.012 |
| 8yxi_BC | 0.911 | 0.775 | 0.281 | 0.009 | 0.192 | 0.491 | +0.088 |
| **8slb_AC** | **0.929** | **0.937** | 0.584 | 0.010 | 0.288 | 0.254 | **+0.296** |

**Yes — better paratope folding goes with a smaller confidence gap.**
Pearson r = **−0.416** (n = 11). Split at paratope lDDT 0.75:

| paratope | n | median confidence gap |
|---|---|---|
| poor (< 0.75) | 6 | **+0.317** |
| good (≥ 0.75) | 5 | **+0.049** |

A 6× difference in median gap. Where the paratope is well folded the correct pose is nearly tied with
the selected one — those are near-misses a better ranker could flip. Where it is poorly folded the
model is confidently, decisively wrong.

### 3a. Deep dive: 8slb_AC — the exception

8slb_AC has the **best paratope in the decoy set** (0.929) *and* a well-folded epitope (0.937), yet a
large gap (+0.296) and a catastrophically wrong selected pose (DockQ 0.010).

| pose | seed / sample | ipTM | DockQ |
|---|---|---|---|
| top-ipTM (what the model picks) | 101 / 1 | 0.584 | **0.010** |
| most confident acceptable | 143 / 0 | 0.288 | 0.254 |
| best DockQ in the pool | 122 / 0 | 0.197 | 0.655 |

**Is the selected pose nearly right?** No. Pose-vs-pose DockQ between the selected pose and the most
confident acceptable pose is **0.012**, and vs the best-DockQ pose **0.009**. These are *different
binding modes*, not a near-miss — the selected pose is not a slightly-off version of the right answer.

**Is it docking to the wrong epitope?** Yes, completely. Antigen residues contacted in the selected
pose vs the native epitope:

| | residues |
|---|---|
| native epitope | 9 |
| predicted epitope (top-ipTM pose) | 6 |
| **overlap** | **0** |
| Jaccard | **0.000** |

**So 8slb_AC is not a folding failure at all.** Both the paratope and the epitope are folded
essentially correctly. The model takes two well-formed surfaces and confidently docks them at the
**wrong patch** of the antigen — zero epitope overlap — and scores that arrangement 0.584 while the
correct site scores 0.288.

**What this means for the hypothesis.** The §2 mechanism (bad binding-site geometry → false
confidence) explains the majority, but **is not the only route to a confident decoy**. 8slb_AC shows a
second, independent failure mode: correct local structure, wrong *site selection*. A method that only
improves fold quality would not fix this case. Note also that the ipTM ordering here is strongly
anti-correlated with correctness — 0.584/0.010, 0.288/0.254, 0.197/0.655 — confidence decreases
monotonically as DockQ increases across these three poses.

### Provenance

- **code:** `scripts/paratope_vs_gap.py` (gap table, correlation, and `--interface` deep dive)
- **data:** `artifacts/paratope_vs_gap.json`, `artifacts/deepdive_8slb_AC.json`

### Caveats

- n = 11; r = −0.416 is a tendency, no significance test run.
- Pose-vs-pose DockQ must be read at the ab–ag interface specifically. Taking `max()` over interfaces
  reported the *other* interface for 8slb and inverted the conclusion to "same binding mode"; fixed.
- Epitope overlap uses OST's native↔model alignment, so it depends on that alignment being right.

---

## 4. Template sweep: what would you have to select folds by? (PROVISIONAL)

**Question.** §2 showed binding-site fold accuracy *correlates* with whether selection succeeds. This
tests it as an **intervention**: template both chains with a model-generated fold, vary only the
criterion used to pick that fold, and see how many confident decoys get rescued.

**Setup.** 7 arms × (15 decoy + 15 good) interfaces, **seeds 101–150 = 250 poses each**, 108 GPU jobs.
Rescue rule identical everywhere: the top interface-`pair_iptm` pose has DockQ ≥ 0.23.
"Goods preserved" is the safety check — a method that rescues decoys by breaking working cases is useless.

### Results

| arm | folds templated | type | decoys rescued | goods preserved |
|---|---|---|---|---|
| native50 | both chains, **native** | oracle | **12/15** | pending¹ |
| **ifacelddt** | both, best **paratope/epitope lDDT** | oracle | **9/15** | 15/15 |
| abonly | **antibody only**, native | oracle | 8/15 | 15/15 |
| globallddt | both, best **whole-chain lDDT** | oracle | 6/15 | 15/15 |
| agonly | **antigen only**, native | oracle | 4/15 | 15/15 |
| ifaceplddt | both, best **paratope/epitope pLDDT** | oracle² | 1/15 | 15/15 |
| **globalplddt** | both, best **whole-chain pLDDT** | **DEPLOYABLE** | **0/15** | 14/15 |

¹ native50's goods were run separately (`safety_out`); scorer now finds them, not yet re-run.
² **Not deployable.** pLDDT is model-internal, but the paratope/epitope regions are defined from the
native complex, which is unknown at inference. Only `globalplddt` requires nothing but the prediction.

### What it shows

1. **Binding-site accuracy is the right thing to select on.** `ifacelddt` (9/15) beats `globallddt`
   (6/15) — identical poses, identical templating machinery, only the criterion differs. It also beats
   native-antibody templating (8/15) and closes most of the gap to the both-chains-native ceiling
   (12/15), using only model-generated folds. This is §2's correlation reproduced as an intervention.
2. **The model's own confidence cannot identify those folds.** Hold the region fixed and swap the
   accuracy measure for the model's confidence over that same region: **9/15 → 1/15**. The information
   needed is present in the pool and pLDDT does not carry it.
3. **Nothing deployable works.** `globalplddt` 0/15, joining `pddag` 0/15 and Exp D 0/15 on the
   corrected set — three independent attempts, same answer.
4. **The antibody side dominates**: templating the native antibody alone (8/15) does twice as much as
   the native antigen alone (4/15), consistent with §2's finding that the paratope, not the epitope,
   separates decoys from goods.

### Status: PROVISIONAL

- **7z2m is incomplete in all four predicted-fold arms** (pools still filling at last scoring) and is
  re-scored under its corrected `A–K`/`B–K` labels afterwards. Counts may move by ±1–2.
- `8gho_AC` and `8q7o_AC` were partial when last scored; their jobs have since finished and they
  re-score automatically (the scorer only skips a full 250-pose pool).
- The `native50` goods column is blank — the ceiling arm's safety check is missing.
- **Full MSA is not in this table.** It is the only other deployable intervention (3/15 on the older
  set) and its predictions exist at seeds 101–150, but its scored results still use the pre-correction
  7z2m label and include 8ux6.

### Provenance

- **code:** `scripts/fold_selection_table.py` (per-pose fold quality, all 4 criteria in one pass) →
  `scripts/build_sweep_templates.py` (template construction) → `scripts/sweep_sched.py` (GPU
  scheduling) → `scripts/score_sweep.py` (one scorer for all arms)
- **data:** `artifacts/sweep_scores.json` (per-arm, per-interface), `artifacts/fold_selection_table.json`
  (per-pose fold quality), `artifacts/sweep_jobs.json` (the 108 jobs)
- **predictions:** `artifacts/sweep_out/<arm>/<pdb>_<arm>/seed_<101-150>/`, plus
  `predictions/{native50_out,abonly_out,agonly_out,safety_out}/` for the native arms

---

## 5. Native-fold templating across the full benchmark (PROVISIONAL)

**Setup change from §1–§4.** Earlier sections used an inherited 15-interface set assembled from a
scores CSV that mixes seed ranges and whose selection criterion was never documented. This section
re-derives the set from scratch: **every FoldBench antibody–antigen interface, all at seeds 101–150,
one scoring protocol.**

| status at seeds 101–150 | count |
|---|---|
| solved — ipTM picks an acceptable pose | 119 |
| **confident decoy** — an acceptable pose exists, ipTM misses it | **28** |
| **generation failure** — no acceptable pose anywhere in 250 | **19** |
| not scored (labelled pair absent even after the isolated-pair fallback) | 6 |

The experiment: template **both chains on the exact native fold** and re-score. Templates carry only
intra-chain geometry (`multichain_mask`), so this fixes *folds* and says nothing about how chains are
arranged — which makes it a clean probe of which bottleneck each interface actually has.

### Outcome classes

Rather than a bare pass/fail, each interface is classified by what templating did or did not repair:

- **RESCUED** — selection now picks an acceptable pose.
- **SELECTION-LIMITED** — coverage is healthy after templating (≥10% of poses acceptable) but ipTM
  still picks a wrong one. Folds fixed, confidence not.
- **COVERAGE-LIMITED** — even with native folds, <10% of poses are acceptable. The docking search,
  not fold quality, is the wall.
- **THIN** — fewer than 5 acceptable poses in the *free* run. These sit within sampling noise of the
  decoy/generation-failure boundary and should not drive conclusions.

### Results

| | rescued | selection-limited | coverage-limited |
|---|---|---|---|
| **Run A — 28 confident decoys** | **19** | 2 | 7 |
| **Run B — 15 generation failures scored** | **4** | 0 | 11 |

Excluding THIN cases, Run A is **14/17**.

**1. Most "selection failures" were really fold failures.** 19 of 28 decoys are rescued, often
dramatically — `8yxi_AC` and `8fbw_AC` go from 7 and 11 acceptable poses to ~250; `8av2_AC` from 1 to
250. Selected DockQ typically moves from ~0.1 to 0.72–0.97.

**2. Only two genuine selection failures survive correct folds:**

| interface | free n_ok | → native n_ok | selected DockQ | best available |
|---|---|---|---|---|
| 8fas_BC | 77 | **203** | 0.175 → **0.173** | 0.994 |
| 7zau_AB | 32 | 5 | 0.004 → 0.011 | 0.787 |

8fas is the repeat-register case from §2's follow-up: 81% of poses correct, and the model still picks
a pose shifted by one `NANP` unit, because a register shift reproduces every local interface signal.

**3. `8pn0_BE` is the cleanest recalibration result.** 155/250 poses were already acceptable in the
free run and ipTM still selected 0.084. With native folds, coverage barely changes (140/250) but the
selected pose jumps to **0.913** — the ranking is fixed without the pool improving. This is §3's
mechanism (fold accuracy drives confidence calibration) isolated from any coverage effect.

**4. "No acceptable pose" does NOT mean docking-limited.** Four interfaces with **zero** acceptable
poses out of 250 gained a full pool from correct folds:

| interface | free best DockQ | → native | selected DockQ |
|---|---|---|---|
| 8sfz_AD | 0.091 | **250 acceptable** | 0.893 |
| 8udz_AD | 0.098 | 118 | 0.580 |
| 8dn7_AC | 0.077 | 52 | 0.495 |
| 8dn7_AF | 0.055 | 47 | 0.416 |

So bad folds can prevent a correct pose from ever being *sampled*, not merely from being *selected*.
The other 11 are unmoved — for those, perfect folds change the ceiling by ~0.01–0.09 and never cross
threshold. `8im0_AB-2` is the extreme: ipTM **0.942**, best DockQ **0.016** free and **0.020**
templated. Maximum confidence, no correct pose reachable even with exact folds.

### Caveats

- **PROVISIONAL:** `7zau_AB` (45/250 poses) and `8oud_AD` (160/250) were still generating at the time
  of writing and re-score automatically. 7zau's selection-limited classification may change.
- **4 interfaces excluded, not failed:** `7yv1_AB/AC` and `8vzo_AB/AC` — template construction failed
  because a query chain matched no native chain, so there is no native arm for them at all.
- **Two label warnings carried through.** `7z2m_AJ` (3 native contact residues) and `8udz_AC` (8) are
  both counted as RESCUED, and both are the crystal-contact-suspect labels from §1. Under 7z2m's
  corrected `A–K`/`B–K` labels it is *not* rescued. Whether to asterisk or exclude these is undecided.
- Thin cases dominate Run B: every one of its 15 interfaces has <5 acceptable poses in the free run by
  definition, so the 4/15 figure rests on small pools.

### Provenance

- **code:** `scripts/abag_all_101_150.py` (classification) → `scripts/build_native_templates.py`
  (native templates for arbitrary pdbs) → `scripts/sweep_sched.py --jobs …` (GPU) →
  `scripts/score_native_arm.py` (scoring) → `scripts/native_arm_summary.py` (failure-mode breakdown)
- **data:** `artifacts/abag_all_101_150.json`, `artifacts/native_arm_scores.json`,
  `artifacts/native_arm_summary.json`
- **environment:** `artifacts/provenance/native_template_arm_*.json` — repo commits, checkpoint hash,
  OpenStructure 2.8.0 / gemmi 0.6.7, the exact OST command, and sha256 of every analysis script

### Gotchas found while running this

- **Chain ids can contain hyphens.** `8im0` has a chain named `B-2`. That broke label parsing
  (`"A-B-2".split("-")`), and Protenix's a3m parser rejects a hyphen in the template hit description —
  the job ran for 8 h producing **zero** seeds while the scheduler recorded it as DONE. Fixed by
  renaming chains to alphanumeric in the template copy, and the watchdog now alerts on any job that
  runs but writes no seeds.
- **Prefer the prediction directory with the most seeds in range, not the first that exists.**
  `predictions/native50_out/` holds seeds 0–49 for 8fbw/8q3j/8yxi while their 101–150 runs are
  elsewhere; taking the first match silently dropped 6 confident decoys from the tally.

---
