# Confident-Decoy Report

Every result carries a `code:` line (the script that produced it) and a `data:` line (the artifact it
can be re-tallied from). A number without both is marked PROVISIONAL.
Paths relative to `/home/ebrahim/confident_decoy_experiments/`. Started 2026-08-01.

---

## 1. The benchmark: 15 confident decoys (SUPERSEDED by §5)

> **Read §5 first for counts.** This 15-interface set was inherited from a scores CSV that mixes seed
> ranges and whose selection criterion was never documented. §5 re-derives the benchmark from all 172
> FoldBench antibody–antigen rows at one seed range under one protocol, giving **26 confident decoys**
> and **19 generation failures**. That is the canonical set.
>
> This section is retained for three reasons: §2–§4 were run on these 15 interfaces; the per-interface
> table below is more detailed than §5's; and it holds the corrected `7z2m` **A–K**/**B–K** rows, which
> §5's set cannot contain because that derivation enumerates FoldBench's labels as given. The 15 and the
> 26 overlap heavily but are **not** nested — do not add or compare them.

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

## 5. Native-fold templating across the full benchmark

**This section defines the canonical interface set.** §1–§4 use an inherited 15-interface set
assembled from a scores CSV that mixes seed ranges and whose selection criterion was never documented.
That set is **superseded**: it is retained because §2–§4's analyses were run on it, but every count
below re-derives the set from scratch — **every FoldBench antibody–antigen interface, all at seeds
101–150, one scoring protocol.** Where the two disagree, this section wins.

| status at seeds 101–150 | count | after exclusions ‡ |
|---|---|---|
| solved — ipTM picks an acceptable pose | 119 | 119 |
| **confident decoy** — an acceptable pose exists, ipTM misses it | 28 | **26** |
| **generation failure** — no acceptable pose anywhere in 250 | 19 | **19** |
| not scored (labelled pair absent even after the isolated-pair fallback) | 6 | 6 |

> **‡ Two interfaces excluded as mislabelled, not as failures.** `7z2m_AJ` (3 native contact residues)
> and `8udz_AC` (8) are the crystal-contact suspects identified in §1's audit: in both cases FoldBench's
> row pairs chains from *different copies* of the same complex, so the "interface" is a lattice contact.
> Both were counted as RESCUED before exclusion, so removing them **lowers** the headline (20/28 → 18/26)
> rather than flattering it.
>
> The exclusion removes the *row*, not the complex, and this matters for 7z2m: its real interfaces
> **A–K** and **B–K** are genuine confident decoys (§1), but the 172-row derivation enumerates
> FoldBench's labels as given and therefore contains only the bad `A–J` row. So 7z2m is absent from the
> canonical 26 despite being a real member of the phenomenon — an undercount, and the reason §1's
> corrected rows are kept rather than deleted. `8udz` remains represented through `8udz_AD`, a separate
> interface the audit did not flag.

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

All 45 interfaces now have a complete native arm at 250 poses (the four that previously failed template
construction — `7yv1_AB/AC`, `8vzo_AB/AC` — were fixed and are included; see Gotchas).

| | rescued | selection-limited | coverage-limited |
|---|---|---|---|
| **Run A — 26 confident decoys** | **18** (69%) | 1 | 7 |
| **Run B — 19 generation failures** | **4** (21%) | 0 | 15 |

Excluding THIN cases, Run A is **14/16**. Every Run B interface is THIN by definition.

**1. Most "selection failures" were really fold failures.** 18 of 26 decoys are rescued, often
dramatically — `8yxi_AC` and `8fbw_AC` go from 7 and 11 acceptable poses to ~250; `8av2_AC` from 1 to
250. Selected DockQ typically moves from ~0.1 to 0.72–0.99. **The single largest determinant of whether
ipTM picks the right pose is whether the binding site was folded correctly.**

**2. Exactly one genuine selection failure survives correct folds.**

| interface | free n_ok | → native n_ok | selected DockQ | best available |
|---|---|---|---|---|
| 8fas_BC | 77 | **203/250 (81%)** | 0.175 → **0.173** | 0.994 |

8fas is the repeat-register case from §2's follow-up: 81% of poses are correct, and the model still
picks a pose shifted by one `NANP` unit, because a register shift reproduces every local interface
signal. This is the only interface in the benchmark where fold quality is demonstrably not the
bottleneck and confidence alone is at fault.

> **Correction.** An earlier draft listed `7zau_AB` as a second selection failure from a partial run
> (5 acceptable poses, selected DockQ 0.011). On the completed run it is **RESCUED** — 23 acceptable
> poses, selected DockQ **0.919**. The provisional number was read while the job was still generating.

**3. `8pn0_BE` is the cleanest recalibration result.** 155/250 poses were already acceptable in the free
run and ipTM still selected 0.084. With native folds the selected pose jumps to **0.913** — ranking
fixed with essentially no change in what was available. This is §3's mechanism (fold accuracy drives
confidence calibration) isolated from any coverage effect.

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

### 5a. Are the 8 unrescued decoys really coverage-starved, or just under-sampled?

Seven of the eight had 1–6 acceptable poses out of 250 (0.4–2.4%). At that rate, "1 acceptable pose"
versus "0 acceptable poses" is close to a coin flip — and that is exactly the line between *confident
decoy* and *generation failure*. Scoring the full free range already on disk (**755 poses**, seeds
0–150, no GPU needed) tests whether the classification is a sampling artifact.

| interface | acceptable, seeds 101–150 | acceptable, seeds 0–150 | best DockQ |
|---|---|---|---|
| 8be1_AC | 1/250 (0.4%) | 2/755 (0.3%) | 0.235 → 0.294 |
| 8fas_BC | 77/250 (30.8%) | 246/755 (32.6%) | 0.987 → 0.987 |
| 8ivx_AC | 4/250 (1.6%) | 19/755 (2.5%) | 0.333 → 0.362 |
| 8oxw_AB | 2/250 (0.8%) | 3/755 (0.4%) | 0.298 → 0.298 |
| 8oxw_AC | 2/250 (0.8%) | 3/755 (0.4%) | 0.275 → 0.275 |
| 8oxx_AC | 1/250 (0.4%) | 1/755 (0.1%) | 0.343 → 0.343 |
| 8ux6_AF | 1/250 (0.4%) | 6/755 (0.8%) | 0.246 → 0.355 |
| 8vui_BC | 6/250 (2.4%) | 6/755 (0.8%) | 0.404 → 0.404 |

**Coverage starvation is real, not an artifact.** Tripling the pool holds the acceptable rate at
0.1–2.5% for all seven thin cases, and none crosses into "solved". The rates are stable, so these
interfaces genuinely almost never produce a correct pose — the classification stands.

### 5b. Does more sampling fix 8ivx and 8oxw? (matched free-vs-native, 755 poses)

§5a deepened the *free* arm only, so those numbers cannot be compared against a 250-pose native arm.
Extending **both** arms to seeds 0–150 makes the comparison matched:

| interface | free acceptable | free selected | native acceptable | native selected | native best |
|---|---|---|---|---|---|
| 8ivx_AB | 1/755 (0.1%) | 0.005 | 3/755 (0.4%) | 0.009 | 0.460 |
| 8ivx_AC | 19/755 (2.5%) | 0.007 | 7/755 (0.9%) | 0.019 | 0.334 |
| 8oxw_AB | 3/755 (0.4%) | 0.019 | **27/745 (3.6%)** | 0.004 | **0.693** |
| 8oxw_AC | 3/755 (0.4%) | 0.009 | **28/745 (3.8%)** | 0.004 | **0.623** |

**8oxw separates generation from ranking cleanly.** Native folds multiply its acceptable poses ~9× and
lift the ceiling from 0.28 to 0.69 — yet selection still lands on **0.004**, with 27 correct poses in
the pool. Fixing the folds fixed the *generation* problem and left the *ranking* problem untouched.
More seeds do not help either: the rate is flat across 755 poses. 8ivx shows no such gain.

Note that `8ivx_AB` — classified a generation failure at 250 poses — has 1 acceptable pose at 755. The
decoy/generation-failure boundary is genuinely porous for thin cases, which is why they are flagged.

### 5c. Which folds does templating actually repair? (interface-local)

The 19 generation failures split 4 rescued / 15 unmoved. An earlier comparison found the rescued group
had *worse* whole-chain folds (median lDDT 0.885) than the unmoved group (0.931) and concluded that
templating helps where there is fold error to correct. **That comparison used whole-chain lDDT, which
8oxw showed is unsafe:** 8oxw's heavy chain scores 0.917 whole-chain but **0.711** across its 14
paratope residues — averaging over 225 residues hid a large error in the only region that matters.

Recomputing both groups interface-locally (lDDT restricted to residues within 5 Å of the partner):

| group | chain | whole-chain lDDT | **interface lDDT** | n chains |
|---|---|---|---|---|
| **rescued** | heavy | 0.892 | **0.689** | 2 |
| **rescued** | light | 0.919 | 0.848 | 2 |
| **rescued** | antigen | 0.842 | 0.848 | 4 |
| unmoved | heavy | 0.931 | **0.819** | 11 |
| unmoved | light | 0.948 | 0.840 | 3 |
| unmoved | antigen | 0.897 | 0.848 | 16 |

The heavy-chain paratope is the discriminating quantity (0.689 rescued vs 0.819 unmoved), consistent
with §2, where the decoy-vs-good paratope gap was +0.312 against a whole-chain gap of only +0.024.

**Two honest limits on this table.** First, **n = 2** on the rescued side — a median over two chains is
an anecdote, not a result; the direction agrees with §2's much larger sample, which is why it is
reported at all. Second, the separation is weaker than the first few rows suggested: the unmoved heavy
chains sit at 0.819, not the ~0.93 that `8dtn` and `8im0` alone imply. There is real overlap.

**What does hold robustly:** interface-local lDDT is lower than whole-chain for *every* group, by
0.03–0.20. Whole-chain fold quality systematically overstates how well the binding site is modelled,
and any analysis resting on it — including the superseded 0.885-vs-0.931 comparison — is measuring the
wrong region.

### Caveats

- **Thin cases dominate Run B:** all 19 interfaces have <5 acceptable poses in the free run by
  definition, so the 4/19 figure rests on small pools. §5a shows the thinness is real rather than a
  sampling artifact, but the pools remain small.
- **Run A's 18/26 includes 4 THIN rescues** (`7ucx_AC`, `8av2_AC`, `8yxi_BC`, `9jbq_AC`, each with 1–4
  free acceptable poses). Excluding all THIN cases gives **14/16**.
- **11 of the 15 unmoved generation failures remain unexplained.** Perfect folds move their ceiling by
  ~0.01–0.09 and never cross threshold. `8im0_AB-2` is the extreme: ipTM **0.942**, best DockQ 0.016
  free and 0.020 templated — maximum confidence, no reachable correct pose even with exact folds. Why
  near-native folds still cannot dock is open.
- **Native templating is not a deployable method.** It requires the answer. It is a probe that
  partitions blame between folding and docking/ranking, nothing more.

### Provenance

- **code:** `scripts/abag_all_101_150.py` (classification) → `scripts/build_native_templates.py`
  (native templates for arbitrary pdbs) → `scripts/sweep_sched.py --jobs …` (GPU) →
  `scripts/score_native_arm.py` (scoring) → `scripts/native_arm_summary.py` (failure-mode breakdown)
- **code (§5a):** `scripts/deep_sample_failures.py` — free arm at seeds 0–150 for the 8 unrescued decoys
- **code (§5b):** `scripts/deep_native_compare.py` — matched free-vs-native at 755 poses, 8ivx + 8oxw
- **code (§5c):** `scripts/genfail_iface_folds.py` — whole-chain vs interface-local lDDT by group
- **code (GPU split):** `scripts/split_seeds_run.sh` — splits one PDB's remaining seeds across GPUs
  (turned 7zau's extension from ~11 h serial into ~2 h)
- **data:** `artifacts/abag_all_101_150.json`, `artifacts/native_arm_scores.json`,
  `artifacts/native_arm_summary.json`, `artifacts/deep_sample_failures.json`,
  `artifacts/deep_native_compare.json`, `artifacts/genfail_iface_folds.json`
- **environment:** `artifacts/provenance/native_template_arm_*.json` — repo commits, checkpoint hash,
  OpenStructure 2.8.0 / gemmi 0.6.7, the exact OST command, and sha256 of every analysis script

### Validation against the reference scorer

The classification above was produced by `scripts/abag_all_101_150.py`, a reimplementation. To check it
is faithful, every interface for which the canonical CSV (written by `eval_ost_foldbench.py` itself)
also holds all 50 seeds at 101–150 was classified both ways and compared:

**125 overlapping interfaces, 125 status agreements, 0 disagreements.**

So the 28 decoys (26 after exclusions) / 19 generation failures are not artifacts of the
reimplementation. The 44 interfaces
the CSV does not cover at these seeds rest on this pipeline alone, but it is one verified against 125
others. This is also the real justification for re-scoring everything rather than reading the 127
already-covered interfaces off the CSV: the duplication bought a cross-check, plus the isolated-pair
fallback gap (8b7h) and per-interface contact counts, which the CSV does not record.

### Gotchas found while running this

- **Chain ids can contain hyphens.** `8im0` has a chain named `B-2`. That broke label parsing
  (`"A-B-2".split("-")`), and Protenix's a3m parser rejects a hyphen in the template hit description —
  the job ran for 8 h producing **zero** seeds while the scheduler recorded it as DONE. Fixed by
  renaming chains to alphanumeric in the template copy, and the watchdog now alerts on any job that
  runs but writes no seeds.
- **Prefer the prediction directory with the most seeds in range, not the first that exists.**
  `predictions/native50_out/` holds seeds 0–49 for 8fbw/8q3j/8yxi while their 101–150 runs are
  elsewhere; taking the first match silently dropped 6 confident decoys from the tally.
- **Sequence-match native chains by containment, not plain ratio.** `8vzo` chain D has 70 of 215
  residues resolved, so `SequenceMatcher.ratio()` scores the *correct* native chain at 0.49 and the
  match was rejected. Taking `max(ratio, longest-common-block / min-length)` fixes it. A query chain
  with genuinely no native counterpart (e.g. `7yv1` q3, 11 residues, unresolved) is now left
  un-templated instead of failing the whole PDB — this is what recovered `7yv1_AB/AC` and `8vzo_AB/AC`,
  which an earlier draft reported as "excluded, template construction failed".
- **A watchdog's status line is not proof a job is running.** The native-arm watchdog kept reporting
  `scoring phase owns the pipeline / scored 195` for 90 minutes after every process had exited. The
  number was a cached file count. Confirm against `/proc` and `nvidia-smi`, not the monitor's own text.

---

## 6. The `dockqdonor` arm: template from the pose that DOCKED correctly (METHODS; running)

**Status: RUNNING.** Relaunched 2026-08-03 17:48 after a false start; 26 jobs × 50 seeds (1300 seeds)
on GPUs 0–3. No results yet. This section was written before any number existed so the protocol could
be reviewed independently of the outcome.

> **This arm has a PRIOR, and it is not encouraging.** A per-PDB version of the same idea, `topdockq`,
> was run on 2026-07-28 and scored **1/14** (only `8ux6_DF`); the companion `topiptm` arm — folds from
> the top-ipTM *decoy* pose — scored **0/5** and was stopped early because the decoy's own fold
> reinforces the decoy. Recorded conclusion: *the best-docked pose does not carry the best folds*
> (fold and dock are decoupled), losing 7 of oracle's 8 rescues — `8bf4` got a 0.723-DockQ template and
> came back at **0.003**. Data: `artifacts/scores/topdockq_scores.json`, `topdockq_picks.json`,
> `topiptm_scores.json`; write-up in `RESULTS.md`.
>
> **I did not find this before building the arm**, and reported to the user that the idea was untested.
> It was in `RESULTS.md` and in the project memory index the whole time. The run below proceeds anyway
> because it differs in three ways — donors chosen **per interface** rather than per PDB (8 of 16
> overlapping donors differ), **26 interfaces** rather than 15 (11 never tested), and the prior run was
> stopped early with `7z2m` never scored — but it is an extension of a mostly-answered question, not a
> new one. Expected landing zone is 1–3 of 26.

### Question

§4's four predicted-fold arms all choose the donor pose by **fold quality** (whole-chain lDDT,
interface lDDT, whole-chain pLDDT, interface pLDDT). None chooses by **DockQ**. This arm asks:

> Take the pose that **docked correctly** but which ipTM ranked low. Feed its folds back as templates.
> Does the model then find the right pose?

The mechanism under test is CDR-H3 induced fit. H3 is the region where our folds are worst (§2, and
the per-residue analysis: framework 0.94 vs H3 0.77), it has no canonical conformations and no MSA
signal, and its conformation is **coupled to the bound pose**. A pose that docked correctly may
therefore hold H3 near its bound conformation — information that a fold-accuracy criterion measured
against the native does not specifically reward.

**Templates carry no docking information.** `multichain_mask` means a template contributes only that
chain's own backbone geometry. Nothing about how the donor arranged its chains is transferred; the
model must still find the pose itself. This is the same property that makes §5's native-fold arm a
clean probe.

### Pre-check: is this redundant with `ifacelddt`? (no)

If the best-DockQ pose were usually also the best-interface-lDDT pose, this arm would re-run an
experiment we already have. Checked first, with no GPU, by comparing the two donors pose-for-pose over
all 250 free-run poses:

| | decoy interfaces (24 chains) | good interfaces (30 chains) |
|---|---|---|
| the two criteria pick the **same** donor | 5 (21%) | 3 (10%) |
| interface lDDT given up by the DockQ donor (median) | **0.012** | 0.008 |
| DockQ given up by the lDDT donor (median) | **0.291** | 0.037 |

They disagree on 79% of decoy chains, and the trade is one-sided: choosing by DockQ costs essentially
nothing in binding-site fold accuracy (0.012 lDDT), while choosing by lDDT costs 0.291 DockQ. The
fold-best pose is often one that docked catastrophically — `8the` A (0.010), `8yxi` C (0.006), `8q3j` A
(0.064), `8slb` C (0.021). So the two donors are near-matched on fold accuracy and differ mainly in
whether their pose docked, which is what makes a rescue difference interpretable.

- **code:** `scripts/dockq_vs_lddt_donor.py` · **data:** `artifacts/dockq_vs_lddt_donor.json`
- Scope: ran on the §1 15-decoy set (24 chains), the set for which `fold_selection_table.json` exists.

### Donor selection rule (exact)

For **each of the 26 confident decoy interfaces** (§5's canonical set, `7z2m_AJ` and `8udz_AC`
excluded):

1. Read the DockQ of that interface for all 250 free-run poses (seeds 101–150 × 5 samples) from the
   cached OST runs. The interface is matched **by chain name** in `dockq_interfaces`, never by
   `dockq[0]`.
2. Keep poses with **DockQ ≥ 0.23**.
3. The donor is the **single pose with the highest DockQ** among those.
4. **Every protein chain of the complex** is templated from that one pose — including chains not in
   the interface under test (e.g. `8slb` chain C when testing A–B), so that all folds come from one
   prediction and only one thing differs from the free run.

One run per **interface**, not per PDB. The six PDBs with two decoy interfaces (`8che`, `8fbw`,
`8oxw`, `8q3j`, `8slb`, `8yxi`) get two separate runs with different donors — e.g. `8yxi_AC` uses
seed 135/sample 2 (DockQ 0.295) and `8yxi_BC` uses seed 140/sample 1 (DockQ 0.538).

> **Two rejected designs, recorded because the first one was built and launched before being caught.**
>
> **Per-chain donors (wrong).** The first build selected a donor per chain, to match how `ifacelddt`
> selects. That is a category error: lDDT and pLDDT have a value *per chain*, but **DockQ is a property
> of an interface** and has no per-chain value. Forcing one required `max()` over every interface a
> chain belonged to, which let an unrelated interface choose a shared chain's folds — when testing
> `8yxi` A–C, chain C's donor had been picked by its score on B–C. It also produced **chimeric**
> template sets in 6 of 20 complexes, mixing folds from different poses into a combination no single
> prediction ever produced. Killed after ~10 min of GPU; no output scored.
>
> **Leaving non-interface chains un-templated (not chosen).** Considered for `8slb` C and similar.
> Rejected because B and C are a paired Fab, so letting C fold freely could drag B, adding a second
> difference from the free run.

### Template construction

Identical mechanics to §5's native arm and README 6.3, only the fold source differs:

1. The donor pose's mmCIF is registered as a **fake-old** template under a fresh tid
   (`release_date 2000-01-01`, so Protenix's date filter accepts it) in `/home/ebrahim/mmcif/`.
2. For each protein chain, a **2-line a3m** points at the matching chain of that tid. The model chain
   is matched to the query **by sequence**, since poses have no stable chain naming.
3. **Two residues of the hit are mutated** (at L/3 and 2L/3) so Protenix treats the row as a template
   rather than an MSA sequence.

Native chain identities are resolved by **`label_asym_id`**, via a CIF-level reader (`native_seqs()`),
not `gemmi.read_structure()`.

Pre-launch verification (all passed): 75 a3m files, each referencing an existing registered mmCIF,
query and hit of equal length, hit differing from the query, and **every job's template paths belonging
to its own interface**; 26 jobs, 26 unique interfaces, 26 unique run names, with an assert in the
builder that fails the build if any interface yields more than one job.

### Generation

Per job, issued by `scripts/sweep_sched.py` (one job per GPU, global queue, seed-level resume):

```bash
CUDA_VISIBLE_DEVICES=<0-3> LAYERNORM_TYPE=openfold OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path artifacts/sweep_tmpl/dockqdonor/<interface>.json \
  --dump_dir artifacts/sweep_out/dockqdonor \
  --seeds 101,...,150 \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 --use_template true
```

Seeds 101–150 × 5 samples = **250 poses per interface**, the same protocol as every other arm.

### Scoring (planned)

Unchanged from `tts_abag/eval/eval_ost_foldbench.py`: one OST call per pose, full native assembly vs
full model, labelled interface extracted **by name**, isolated-pair fallback only on failure.

```bash
ost compare-structures -m <pred>.cif -r <native>.cif -o <out>.json \
    --fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq
```

Rescue rule identical to every other arm: the top interface-`pair_iptm` pose has **DockQ ≥ 0.23**.

**`ifacelddt` will be re-scored on these same 26 interfaces before the two are tabulated together.**
Its published 9/15 is on the §1 set; comparing 26 against 15 would not be like-for-like.

### Caveats (read before interpreting any result)

- **This is an ORACLE, not a deployable method.** Selecting the donor by DockQ requires the native
  structure. Like `ifacelddt` and `native50`, it measures a ceiling and attributes blame — it is not a
  method that could be shipped.
- **10 of 26 donor pools are THIN** (<5 acceptable poses to choose from): `7ucx_AC`, `8av2_AC`,
  `8be1_AC`, `8oxx_AC`, `8ux6_AF`, `9jbq_AC` (1 each), `8oxw_AB`, `8oxw_AC` (2), `8ivx_AC`,
  `8yxi_BC` (4). For those the "best DockQ pose" is barely a selection at all, and several donors are
  only marginally acceptable (`8ux6_AF` 0.246, `8be1_AC` 0.235). Median donor DockQ across all 26 is
  0.51.
- **A null result is ambiguous.** If this does not beat `ifacelddt`, it could mean the correctly-docked
  pose carries no extra conformational information, *or* that whatever it carries is destroyed by
  templating a fold whose sub-Å accuracy matters (the mechanism behind `self-template fails`, where
  model-derived templates were net-harmful and only exact native folds worked).

### Queued next (launched 2026-08-03 17:52): extending `ifacelddt` and `globallddt` to the 26

Both arms exist only on the §1 15-decoy set. Extending them needs per-pose, per-chain fold quality for
the 10 PDBs absent from `fold_selection_table.json` (`7ucx`, `7zau`, `8be1`, `8ivx`, `8oxw`, `8oxx`,
`8pn0`, `8ux6`, `8vui`, `9jbq`). That computation is CPU/OST only and is running now alongside the GPU
job — `scripts/fold_selection_table.py --pdbs ...` → `artifacts/fold_selection_table.json`, log at
`artifacts/decoy_logs/foldtable_ext.log`.

`interfaces_by_pdb()` in that script now takes the **union** of the original 15-decoy set and §5's
canonical 26, so extending an arm cannot silently drop the older interfaces (the two sets are not
nested — see §1's banner).

**Open design decision, not yet made:** whether the extended arms run per **interface** (26 jobs/arm;
interface residues defined for the interface under test; comparable to `dockqdonor`) or per **PDB**
(20 jobs/arm; interface residues = union over a chain's interfaces; reproduces the design that gave
9/15). Only the 6 two-interface PDBs differ. Not started — the GPUs are occupied by `dockqdonor`.

### Provenance

- **code:** `scripts/dockq_vs_lddt_donor.py` (redundancy pre-check, no GPU) →
  `scripts/build_dockq_templates.py` (donor selection + template construction) →
  `scripts/sweep_sched.py --jobs artifacts/dockq_arm_jobs.json --out artifacts/sweep_out/dockqdonor
  --logs artifacts/dockq_arm_logs --status artifacts/dockq_arm_status.json` (generation)
- **data:** `artifacts/dockq_arm_donors.json` — per interface: donor seed/sample, donor DockQ, pool
  size, per-chain model↔native chain mapping, template tid
- **data:** `artifacts/dockq_arm_jobs.json` (26 jobs), `artifacts/dockq_arm_status.json` (live status)
- **templates:** `artifacts/sweep_tmpl/dockqdonor/<interface>.json`,
  `artifacts/sweep_a3m/dockqdonor_<interface>_q<N>.a3m`
- **predictions:** `artifacts/sweep_out/dockqdonor/<pdb>_<chains>_dockqdonor/seed_<101-150>/`
- **logs:** `artifacts/dockq_arm_logs/`, scheduler log `artifacts/decoy_logs/dockq_arm.log`

### Gotchas found while building this

- **A per-chain rule cannot be applied to a per-interface quantity.** See the rejected designs above.
  This is the same `max()`-over-interfaces shape that once inverted the 8slb_AC conclusion.
- **`label_asym_id` again.** The first build resolved native chains with `gemmi.read_structure()`
  (auth naming) and so matched **no** labelled interface for 8bf4, 8che, 8fbw, 8slb, 8the and 8vui —
  six of twenty complexes would have run with **every chain un-templated** while being reported as a
  template arm. Visible only because whole complexes printed as `free-no-interface`.
- **An edited loop can leave its own tail behind.** After rewriting the builder's loop, the old tail
  still appended a second job per iteration — 52 jobs for 26 interfaces, each overwriting the other's
  template JSON. Now caught by an assert on unique interfaces per job.
- **The scheduler's log path keyed on `pdb`, not on the job.** With one job per interface, `8slb_AB`
  and `8slb_AC` wrote to the same log and one silently overwrote the other. Output directories and
  seed-level resume were unaffected (they key on `job['name']`). Fixed for subsequent runs.

---
