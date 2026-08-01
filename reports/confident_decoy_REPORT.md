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
    --fault-tolerant --min-pep-length 4 --dockq
```

`pair_iptm` is read from each pose's `_summary_confidence_sample_*.json` at the OST-mapped model chains.

### The set

`contacts` = residues within 5 Å across the native interface (sanity check that the interface is real).
`n_ok` = poses with DockQ ≥ 0.23. `top ipTM` = confidence of the #1-ranked pose. `sel DockQ` = that
pose's DockQ (decides the verdict). `best DockQ` = best pose in the pool — the ceiling perfect
selection would reach.

| interface | chains | contacts | n_ok | top ipTM | sel DockQ | best DockQ |
|---|---|---|---|---|---|---|
| 7z2m_AK † | A–K | 19 | 11 | 0.370 | 0.051 | 0.433 |
| 7z2m_BK † | B–K | 31 | 77 | 0.400 | 0.034 | 0.384 |
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
> Scored with **copy-optimal mapping**: the native is subset to the two interface chains and OST is given
> the full model, so it matches the best-corresponding predicted copy. Without this, one fixed global
> mapping ties each native copy to an arbitrary predicted copy and symmetry-equivalent interfaces
> disagree (0.027 vs 0.236 for the same interaction).
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
| antigen whole-chain lDDT | 0.902 | 0.896 | 0.896 | −0.006 |
| epitope lDDT | 0.879 | 0.898 | 0.906 | +0.027 |
| antibody whole-chain pLDDT | 0.869 | 0.854 | 0.911 | +0.042 |
| paratope pLDDT | 0.792 | 0.691 | 0.930 | +0.138 |
| antigen whole-chain pLDDT | 0.849 | 0.820 | 0.875 | +0.026 |
| epitope pLDDT | 0.870 | 0.773 | 0.929 | +0.059 |

**The hypothesis is supported, in a specific and narrow form:**

1. **Paratope accuracy separates the two sets; whole-chain accuracy does not.** The paratope gap
   (+0.312) is **13× the antibody whole-chain gap** (+0.024). Decoy antibodies are globally well folded
   (0.915) but their binding loops are not (0.614). This is exactly the localisation the hypothesis
   predicts, and it rules out "goods are simply easier targets".
2. **It is an ANTIBODY effect, not an antigen one.** Epitope gap is +0.027 and antigen whole-chain is
   −0.006 — i.e. no antigen-side difference at all. The binding-site inaccuracy that tracks false
   confidence sits on the paratope.
3. **Within decoys, the correctly-docked pose has a better paratope** (0.760 vs 0.614). Paratope
   accuracy and docking correctness move together pose-by-pose, not just target-by-target.
4. **The model is more confident about the paratope when it is wrong.** Paratope pLDDT is *higher* at
   the top-ipTM (wrong) pose than at the best-DockQ (right) pose, 0.792 vs 0.691. Confidence is
   anti-correlated with correctness exactly where the hypothesis says the error originates.

### Per-interface paratope lDDT

| decoy | whole-chain | paratope @ top-ipTM | paratope @ best-DockQ |
|---|---|---|---|
| 8yxi_AC | 0.891 | 0.458 | 0.466 |
| 8q3j_AC | 0.876 | 0.525 | 0.608 |
| 8the_AB | 0.854 | 0.528 | 0.526 |
| 8aok_AB | 0.843 | 0.565 | 0.702 |
| 8av2_AC | 0.801 | 0.608 | 0.752 |
| 8slb_AB | 0.915 | 0.614 | 0.769 |
| 7z2m_AK | 0.937 | 0.751 | 0.760 |
| 7z2m_BK | 0.931 | 0.818 | 0.831 |
| 8bf4_AB | 0.917 | 0.830 | 0.859 |
| 8yxi_BC | 0.919 | 0.912 | 0.908 |
| 8slb_AC | 0.960 | 0.928 | 0.932 |

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
| 8qz3_AC | 0.957 | 0.940 |
| 7xld_AB | 0.958 | 0.965 |
| 8pih_AB | 0.970 | 0.977 |
| 8q7o_AC | 0.966 | 0.978 |
| 8w90_AB | 0.960 | 0.979 |

**Overlap is real, so this is a tendency, not a clean separator.** 3 of 11 decoys have good paratopes
(8yxi_BC 0.912, 8slb_AC 0.928) and one good has a poor one (9fww_AB 0.516). Medians differ sharply;
the distributions touch. n = 11 vs 14, no significance test run.

### Provenance

- **code:** `scripts/fold_quality.py` (decoys + goods, both poses, lDDT + pLDDT)
- **code:** `scripts/decoy_set.py --set goods` — goods DockQ at seeds 101–150
- **data:** `artifacts/fold_quality.json` — per-interface, per-pose, all 8 metrics
- **data:** `artifacts/goods_set_101_150.json` — goods selection verdicts

### Caveats

- Excluded 5 peptide-antigen interfaces; conclusions cover domain antigens only.
- Earlier version of this analysis handed OST the full model against a one-chain native, which mapped
  the wrong predicted chain and returned lDDT ≈ 0.004. Fixed by matching the model chain by sequence and
  comparing single-chain to single-chain. **Absurd values mean suspect the code first.**
- Correlational. It does not establish that fixing the paratope fixes selection — though the native-template
  experiments (rescue 14/15 by collapsing the decoy's ipTM) point the same way.

---
