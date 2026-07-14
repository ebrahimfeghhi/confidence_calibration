# DockQ: what is ground truth, and what our GPU code is for

_Settled 2026-07-13. **Read this before touching anything DockQ-related.** It exists so nobody has
to rediscover any of it._

## The one-paragraph version

**OST is ground truth.** `eval/eval_ost_foldbench.py` scores a prediction against the crystal
structure, and it is what FoldBench uses. Every model — ESMFold2, Protenix, anything new — must be
scored with it. **Our GPU DockQ (`steer/experiments/dockq_torch.py`) is for CLUSTERING only**:
pose-vs-pose similarity. It has been **verified to reproduce OST exactly** (below). Do not use it
for ground truth, and do not try to make it do ground truth — that is a much harder problem and it
is already solved.

## Why the split, in one idea

To compute DockQ between two structures you must know **which residue corresponds to which**.
Everything else (contacts, RMSDs, the formula) is trivial arithmetic. That correspondence is the
whole difficulty:

| | residue correspondence |
|---|---|
| **pose vs pose** (our clustering) | **FREE.** Two predictions of the same sequence by the same model have exactly the same residues, in the same order. Index i = index i. |
| **pose vs crystal** (ground truth) | **HARD.** The crystal has unresolved residues (gaps), antibody chains use non-sequential Kabat numbering with insertion codes, chain names differ (crystal H/L/A vs model A/B/C), and the model predicts residues the crystal does not contain. You must align them. |

**Every DockQ bug in this project has been in that alignment, or in chain naming — never in the
DockQ math.** Which is why OST does ground truth and we do not.

## Verification: our GPU DockQ reproduces OST exactly

Run OST pose-vs-pose (one prediction as the "reference", another as the "model") and compare to
`dockq_torch.py`. Same complex, so no alignment is involved anywhere — exactly the regime our
clustering lives in. 8ts0 A-C, pose pairs spanning the DockQ range:

| pose pair | our GPU | OST | diff |
|---|:--:|:--:|:--:|
| (0, 1) | 0.4359 | 0.436 | 0.0001 |
| (0, 500) | 0.0285 | 0.029 | 0.0005 |
| (10, 2000) | 0.5351 | 0.535 | 0.0001 |
| (100, 3000) | 0.5682 | 0.568 | 0.0002 |
| (5, 40) | 0.0326 | 0.033 | 0.0004 |

Every difference is <= 0.0005 — that is **OST rounding to 3 decimals**. There is no real
disagreement. (`dockq_torch.py` also matches the reference DockQ python package on **7,354,375
pairs**, max error 1e-6.)

## The traps, so nobody falls in them again

**1. Chain names: the CSV's `chain1`/`chain2` are NATIVE chain ids, not model ones.**
The `chain_mapping` column (ref -> model) translates. On multi-copy assemblies the mapping
**changes from pose to pose** — OST maps the native onto whichever model copy fits, so 8oni_AE uses
model chains A-E, A-F, B-E or B-F depending on the seed. Using the native letters directly on a
model file scores **the wrong pair of molecules**. This corrupted 17 of 48 interfaces before it was
caught. `dockq_torch.pose_list()` now selects the chain pair PER POSE.

**2. Native chains are named by `label_asym_id`, not `auth_asym_id`.**
OST (and therefore every CSV) uses label ids; gemmi and the DockQ python package default to auth.
On 8ts0 the heavy chain is auth **H** but label **A**, and the antigen is auth **A** but label
**C** — so an auth-based lookup silently grabs the wrong molecule, or a chain that does not exist.

**3. `Protenix_dockq.csv` (the OLD file) is WRONG. Do not use it.**
It has no `dockq_status` column: it was never produced by `eval_ost_foldbench.py`. Its values are
wrong in **both** directions — it reported **0.000** for 8ts0 where Protenix really scores
**0.950**, and **0.675** for 8q3j where Protenix really picks **0.035**.
**Use `results/csv_files/dockq_ost_protenix.csv`** (12,250 rows, all `status=ok`, scored by the
same OST pipeline as ESMFold2).

**4. ESMFold2's ground truth is SOUND.** 100% of rows are `dockq_status=ok`; the buggy
`ok_isolated` fallback in the eval script (which subsets the MODEL by NATIVE chain names — trap #1
again, inside the eval script) never fired for our interfaces.

## Where the numbers live

| file | what |
|---|---|
| `results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv` | **ESMFold2 ground truth** (OST). ~374k poses. |
| `results/csv_files/dockq_ost_protenix.csv` | **Protenix ground truth** (OST). 12,250 poses. |
| `results/csv_files/Protenix_dockq.csv` | **DO NOT USE.** Wrong; kept only for provenance. |
| `results/dockq_full/dockq_<iface>.npy` | pose x pose DockQ matrices (our GPU code) — clustering only. |
| `steer/experiments/dockq_torch.py` | the GPU DockQ. Clustering only. Verified against OST. |
| `eval/eval_ost_foldbench.py` | OST. **Ground truth for every model.** `OST_WORKERS=N` bounds workers. |

## Running OST on a new model

```bash
# Shared box: pin the tree, single-thread the children, and CHECK MEASURED usage at 10s.
taskset -c 0-63 env OST_WORKERS=20 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 19 .venv/bin/python eval/eval_ost_foldbench.py \
    --pred_dir <predictions> --out_csv results/csv_files/dockq_ost_<model>.csv
```
`OST_WORKERS=20` lands at ~85 processes (each worker forks a multithreaded `ost` subprocess — the
worker count is NOT the process count; `OST_WORKERS=128` once produced ~600 processes and drove
load to 774 on a 288-core shared box). ~6,750 poses takes ~3 minutes.

## What was deleted, and why

`steer/experiments/dockq_vs_native.py` — an attempt to reimplement pose-vs-CRYSTAL DockQ on GPU. It
got as far as reproducing OST exactly on 8ts0, then broke on 8vui's Kabat numbering. It was never
needed: OST already does this, correctly. Deleted rather than maintained half-working.

## Two more traps, found 2026-07-14 (both nearly manufactured a fake result)

**5. gemmi 0.6.7 silently returns ZERO MODELS for ESMFold2's CIFs.** gemmi 0.7.5 reads both models'
files. The Protenix venv has 0.6.7; `~/tts_abag/.venv-gpu` has 0.7.5. **Build both sides of any
comparison with the SAME parser** — using two different ones is precisely how a fake result gets
made. Both `results/dockq_protenix/` and `results/dockq_esm250/` were built with `.venv-gpu`.

**6. Verifying against OST requires an EXPLICIT `-c REF:MDL` chain map.** Left to itself, OST
re-derives its own chain correspondence between the two poses, and on a multi-copy assembly it will
happily pair a DIFFERENT copy than your matrix used. Then OST and you are scoring two different
comparisons and the diff is meaningless. This produced a phantom **0.2018 "mismatch"** on 8pmz_AB
that vanished to **0.0004** once OST was pinned to our pairing. (Also: OST reports `dockq` as a
**list** parallel to `dockq_interfaces`; there is no `dockq_scores` key. Looking for one produced
12/12 phantom "OST failures".) See `steer/experiments/verify_protenix_dockq.py`.

## Protenix pose-vs-pose matrices (new, 2026-07-14)

| file | what |
|---|---|
| `results/dockq_protenix/dockq_<iface>.npy` | 165 interfaces, 250 poses each (50 seeds x 5 samples) |
| `results/dockq_esm250/dockq_<iface>.npy` | the SAME interfaces from ESMFold2, subsampled to 250 poses with the same rng seed, so the two are depth-matched pose-for-pose |
| `steer/experiments/dockq_protenix.py` | the builder. Imports `dockq_torch.dockq_matrix` — adds NO DockQ formula. |
| `steer/experiments/dockq_esm_matched.py` | ditto for ESM |

**Certified against OST**: 46/46 pose pairs across 12 random interfaces, worst diff **0.0005**
(= OST's 3-decimal rounding). **Known limit:** interfaces carrying modified residues deviate by up
to **0.0045** — e.g. 8vvl_AC, whose chain C has a PCA (pyroglutamate); our atom-level code counts
its atoms, OST resolves it through its chemical-component dictionary. Negligible for
cohesion/clustering statistics (tau = 0.35), but it is not zero.

**Skip-and-log guard:** `dockq_torch` asserts a consistent atom count per chain across poses. Some
multi-copy assemblies map the same chain letter onto differently-sized copies (8a67_AG, 8sfz_BF).
Those interfaces are SKIPPED and logged rather than scored wrongly — that assertion is what caught
the 17/48 chain-mapping corruption in the first place.
