# Train / test split — confident-decoy benchmark

**Fixed 2026-08-14.** Do not re-derive: this file is the split. Any change is a new split and
invalidates comparisons made against the old one.

## How it was built

- **Unit of assignment is the COMPLEX, never the interface.** 14 of 33 complexes contribute two
  interfaces; splitting one across train/test would leak (same antigen, same antibody, usually the
  same decoy surface). Three complexes (8ivx, 8oxx, 8udz) have one interface in *each* category --
  they are still assigned as whole complexes, so a complex never appears on both sides.

- **All numbers below are from seeds 101-150 only** (50 seeds x 5 samples = 250 poses per
  interface), the canonical free pool. Verified: `decoy_set.py:25` hardcodes `range(101, 151)`, and
  the OST caches contain exactly seeds 101-150 x samples 0-4.

- **The 6 already-examined complexes are forced into TRAIN** (8che, 9jbq, 8av2, 8slb, 8aok, 8fas) --
  method development happened on them, so they cannot test anything.

- **Test >= train in interface count, in each category separately**, and balanced on median residue
  count, median best-DockQ, fraction 'hard', and fraction native-rescuable.

## Definitions

| term | meaning |
|---|---|
| `sel` | DockQ of the pose with the highest interface pair_iptm (the label-free choice) |
| `best` | best DockQ among the 250 poses (oracle ceiling for free sampling) |
| `n_acc` | poses with DockQ >= 0.23 |
| `native50` | DockQ selected when templating from the NATIVE structure -- the rescuability ceiling |
| confident decoy | a good pose exists (`n_acc` >= 1) but ipTM selects a bad one |
| no acceptable pose | no pose reaches 0.23 at all (`n_acc` = 0) |
| hard | `best` < 0.35 OR `n_acc` <= 4 |

## Confident decoys

### TRAIN — 14 interfaces (median 942 res, median best 0.488, hard 6/14, native-rescuable 10/14)

| interface | pdb | resid | sel | best | n_acc | native50 |
|---|---|---|---|---|---|---|
| 7zau_AB | 7zau | 2520 | 0.004 | 0.435 | 32 | 0.919 |
| 8udz_AC | 8udz | 1624 | 0.005 | 0.239 | 1 | 0.796 |
| 8the_AB | 8the | 1230 | 0.113 | 0.514 | 62 | 0.392 |
| 8oxw_AB | 8oxw | 1134 | 0.019 | 0.298 | 2 | 0.004 |
| 8oxw_AC | 8oxw | 1134 | 0.009 | 0.275 | 2 | 0.004 |
| 8ivx_AC | 8ivx | 1029 | 0.007 | 0.333 | 4 | 0.019 |
| 8che_AE \* | 8che | 942 | 0.125 | 0.438 | 25 | 0.959 |
| 8che_BE \* | 8che | 942 | 0.087 | 0.529 | 24 | 0.953 |
| 8slb_AB \* | 8slb | 743 | 0.012 | 0.462 | 65 | 0.907 |
| 8slb_AC \* | 8slb | 743 | 0.010 | 0.655 | 41 | 0.875 |
| 8av2_AC \* | 8av2 | 678 | 0.092 | 0.612 | 1 | 0.878 |
| 9jbq_AC \* | 9jbq | 573 | 0.041 | 0.584 | 1 | 0.990 |
| 8fas_BC \* | 8fas | 462 | 0.175 | 0.987 | 77 | 0.173 |
| 8aok_AB \* | 8aok | 245 | 0.112 | 0.695 | 109 | 0.739 |

### TEST — 14 interfaces (median 942 res, median best 0.472, hard 6/14, native-rescuable 10/14)

| interface | pdb | resid | sel | best | n_acc | native50 |
|---|---|---|---|---|---|---|
| 7z2m_AJ | 7z2m | 2244 | 0.039 | 0.608 | 47 | 0.436 |
| 8pn0_BE | 8pn0 | 1744 | 0.084 | 0.741 | 155 | 0.913 |
| 8bf4_AB | 8bf4 | 1554 | 0.014 | 0.723 | 22 | 0.718 |
| 8be1_AC | 8be1 | 1286 | 0.022 | 0.235 | 1 | 0.061 |
| 8q3j_AB | 8q3j | 1232 | 0.035 | 0.705 | 24 | 0.930 |
| 8q3j_AC | 8q3j | 1232 | 0.036 | 0.531 | 19 | 0.826 |
| 8ux6_AF | 8ux6 | 956 | 0.157 | 0.246 | 1 | 0.086 |
| 8fbw_AC | 8fbw | 928 | 0.170 | 0.434 | 11 | 0.780 |
| 8fbw_BC | 8fbw | 928 | 0.210 | 0.510 | 116 | 0.742 |
| 8oxx_AC | 8oxx | 899 | 0.003 | 0.343 | 1 | 0.039 |
| 8yxi_AC | 8yxi | 781 | 0.006 | 0.295 | 7 | 0.804 |
| 8yxi_BC | 8yxi | 781 | 0.009 | 0.538 | 4 | 0.970 |
| 8vui_BC | 8vui | 668 | 0.034 | 0.404 | 6 | 0.023 |
| 7ucx_AC | 7ucx | 462 | 0.155 | 0.269 | 1 | 0.871 |

## No acceptable pose

### TRAIN — 9 interfaces (median 1006 res, median best 0.091, hard 9/9, native-rescuable 2/9)

| interface | pdb | resid | sel | best | n_acc | native50 |
|---|---|---|---|---|---|---|
| 8udz_AD | 8udz | 1624 | 0.007 | 0.098 | 0 | 0.580 |
| 8sfz_AD | 8sfz | 1194 | 0.061 | 0.091 | 0 | 0.893 |
| 8ivx_AB | 8ivx | 1029 | 0.005 | 0.144 | 0 | 0.009 |
| 8vzo_AB | 8vzo | 1006 | 0.047 | 0.107 | 0 | 0.059 |
| 8vzo_AC | 8vzo | 1006 | 0.021 | 0.083 | 0 | 0.028 |
| 8a44_AC | 8a44 | 792 | 0.019 | 0.038 | 0 | 0.019 |
| 8a44_AD | 8a44 | 792 | 0.017 | 0.019 | 0 | 0.016 |
| 8z8v_AB | 8z8v | 708 | 0.007 | 0.084 | 0 | 0.027 |
| 8dtn_AB | 8dtn | 588 | 0.027 | 0.136 | 0 | 0.026 |

### TEST — 10 interfaces (median 1052 res, median best 0.098, hard 10/10, native-rescuable 2/10)

| interface | pdb | resid | sel | best | n_acc | native50 |
|---|---|---|---|---|---|---|
| 8oud_AD | 8oud | 2001 | 0.005 | 0.157 | 0 | 0.005 |
| 8dn7_AC | 8dn7 | 1550 | 0.025 | 0.077 | 0 | 0.495 |
| 8dn7_AF | 8dn7 | 1550 | 0.014 | 0.055 | 0 | 0.416 |
| 8uky_AC | 8uky | 1204 | 0.032 | 0.044 | 0 | 0.018 |
| 8uky_BC | 8uky | 1204 | 0.023 | 0.054 | 0 | 0.022 |
| 8oxx_AB | 8oxx | 899 | 0.004 | 0.162 | 0 | 0.014 |
| 8a67_AD | 8a67 | 744 | 0.045 | 0.157 | 0 | 0.073 |
| 7yv1_AB | 7yv1 | 629 | 0.082 | 0.119 | 0 | 0.063 |
| 7yv1_AC | 7yv1 | 629 | 0.086 | 0.211 | 0 | 0.030 |
| 8im0_AB-2 | 8im0 | 362 | 0.013 | 0.016 | 0 | 0.013 |

\* = already examined during method development (forced to train).

## Complex assignment

**TRAIN (16)**: 7zau, 8a44, 8aok, 8av2, 8che, 8dtn, 8fas, 8ivx, 8oxw, 8sfz, 8slb, 8the, 8udz, 8vzo, 8z8v, 9jbq

**TEST (17)**: 7ucx, 7yv1, 7z2m, 8a67, 8be1, 8bf4, 8dn7, 8fbw, 8im0, 8oud, 8oxx, 8pn0, 8q3j, 8uky, 8ux6, 8vui, 8yxi

## Caveats

- **8fas_BC should be excluded from any analysis.** Its antigen is the NANP5 repeat peptide and
  OST's aligner slides by one repeat, so its DockQ values do not mean what they appear to (a pose
  0.55 A from the crystal scores 0.174). It sits in train only because it was examined.

- **The no-acceptable-pose category is near-unrescuable by construction**: only 4 of 19 clear 0.23
  even when templated from the NATIVE structure (8sfz_AD 0.893, 8udz_AD 0.580, 8dn7_AC 0.495,
  8dn7_AF 0.416). Expect near-zero success for any method; the signal is which of those 4 move.

- **Test is the more expensive half** (~700 h/arm vs ~450 at 4 GPUs): it holds 7z2m (2244 res),
  8oud (2001), 8pn0 (1744), 8bf4 (1554), 8dn7 x2 (1550). Runtime scales ~(residues)^2.5.

