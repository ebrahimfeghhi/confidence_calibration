# HANDOFF — 250-donor × 10-seed sweeps on 4 complexes, MSA on then MSA off

**Written 2026-08-10 on naboo (the origin box). Target: alderaan (7 free GPUs).** Assumes alderaan has
the same data as naboo (`foldbench_abag/` prep + predictions, `$HOME/mmcif`, `$HOME/common`, checkpoint). Verify in step 2 before
building anything — alderaan was missing `artifacts/` metadata and the donor release-date entries last
night, so do not assume.

---

## Why this experiment

Everything in the confident-decoy project currently rests on **one complex**. On 8che the full sweep —
template from all 250 free poses, 10 seeds each, select the highest-ipTM pose over the whole pool —
rescued **both** interfaces with no labels:

| 8che | free (2500 seeds) | `dockqdonor` (oracle donor, 50 seeds) | **250×10 sweep, pooled ipTM** |
|---|---|---|---|
| BE | 0.064 | 0.958 | **0.924** |
| AE | 0.115 | 0.121 ✗ | **0.938** |

On AE the oracle donor rule *fails* (it picks donor `112_2`, rank 72 of 250) while the label-free sweep
finds `137_3` (rank 1) and returns 0.938. The sweep never has to *predict* which donor is good — it
only has to contain it and let ipTM recognise the output, which is the job ipTM can actually do
(donor-ipTM vs arm-result correlation is +0.03; output ipTM separates cleanly).

**This handoff replicates that on 4 more complexes, with and without the MSA.** Removing the MSA is
worth testing at scale because on the single-donor arms it was strongly bimodal — `8fas_BC` went from
27/50 seeds acceptable and ipTM-selected DockQ 0.169 to **50/50 and 0.982** — while other interfaces
collapsed entirely.

## The 4 complexes and why each

| pdb | interfaces | free selected | best in free pool | `dockqdonor` | `native50` | why it is here |
|---|---|---|---|---|---|---|
| **8fas** | BC | 0.175 | **0.987** | 0.169 ✗ | 0.173 ✗ | Nothing has ever rescued it, yet a near-perfect pose sits in the pool. The hardest selection case we have. |
| **9jbq** | AC | 0.041 | 0.584 | 0.037 ✗ | **0.990** ✓ | Native templating works, so a reachable answer exists — but no label-free *or* oracle donor criterion has found it. |
| **8av2** | AC | 0.092 | 0.612 | **0.672** ✓ | 0.878 ✓ | A case where single-donor templating already works: does the sweep reproduce it? |
| **8slb** | AB + AC | 0.012 / 0.010 | 0.462 / 0.655 | **0.734** / **0.887** ✓ | 0.907 / 0.875 ✓ | Two interfaces from one run, like 8che — tests whether one donor is best for both. |

5 interfaces from 4 runs. Spread deliberately across "nothing works", "only native works", and
"templating already works".

## Cost

250 donors × 10 seeds × 5 samples = 2500 seeds / 12,500 poses per run. Runtime scales steeply with
complex size (fitted exponent ~2.5 from timed runs; 8che at 942 tokens = 13 min per 10-seed job).

| run | single-GPU | on 7 GPUs |
|---|---|---|
| 8fas | 9.3 h | 1.3 h |
| 9jbq | 15.8 h | 2.3 h |
| 8av2 | 24.0 h | 3.4 h |
| 8slb | 30.1 h | 4.3 h |
| **MSA-on total** | 79 h | **~11 h** |
| **MSA-off total** | 79 h | **~11 h** |

The 250 jobs within a run are independent, so `sweep_sched.py` fills all 7 GPUs from a global queue.

---

## 1. Get the code

```bash
cd <tts_abag>                      # github.com/ebrahimfeghhi/tts_abag
git fetch && git checkout reorg/declutter && git pull
```

## 2. Verify prerequisites BEFORE building

```bash
python confident_decoy_experiments/paths.py     # ARTIFACTS_DIR / PREDICTIONS_DIR must say "ok"
for p in 8fas 9jbq 8av2 8slb; do
  echo -n "$p json: ";  ls /home/ebrahim/Protenix/foldbench_abag/jsons/$p-final-updated.json 2>/dev/null | wc -l
  echo -n "$p msa:  ";  du -sh /home/ebrahim/Protenix/foldbench_abag/prep/$p/msa 2>/dev/null | cut -f1
  echo -n "$p free seeds 101-150: "; ls -d /home/ebrahim/Protenix/foldbench_abag/predictions_template/$p/seed_1[0-4][0-9] \
      /home/ebrahim/Protenix/foldbench_abag/predictions_template/$p/seed_150 2>/dev/null | wc -l
done
ls -d /home/ebrahim/checkpoint $HOME/mmcif $HOME/common/release_date_cache.json
```

Each pdb needs **51 free seed dirs** (101–150 plus seed_150 counted twice by that glob — 50 distinct is
correct) and a non-empty `msa/`. On naboo the payload is msa 26–87 MB and predictions 260–450 MB per pdb, **1.7 GB for all four**. If
any of it is absent on alderaan, say so — it has to be copied from naboo, not regenerated.

**If anything is missing, stop and say so — do not regenerate it.** The free predictions are the donor
source; regenerating them would produce a different pool and the run would not be comparable to 8che.

## 3. Build the MSA-on arms

Template ids must be **exactly 4 characters** — Protenix raises `Invalid hit name format` otherwise and
logs `Found 0 templates`, silently producing an untemplated run. Use one unused prefix per pdb:

```bash
cd confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
$PY scripts/build_allpose_generic.py --pdb 8fas --run allpose8fas10 --tid-prefix f --seeds 10
$PY scripts/build_allpose_generic.py --pdb 9jbq --run allpose9jbq10 --tid-prefix j --seeds 10
$PY scripts/build_allpose_generic.py --pdb 8av2 --run allpose8av210 --tid-prefix e --seeds 10
$PY scripts/build_allpose_generic.py --pdb 8slb --run allpose8slb10 --tid-prefix k --seeds 10
```

Prefixes `f j e k` are unused on naboo (`p`=8che, `b`=8bf4, `d`=dockqdonor, `h`=hillclimb are
taken). The builder asserts each id is 4 chars and not already in `$HOME/mmcif`; if it aborts on a
collision, pick another from `m n r x` and say which you used.

Each build prints `BUILT 250 donor jobs x 10 seeds = 2500 seeds`. Anything less means free poses are
missing — stop.

## 4. Preflight, then run each MSA-on arm

```bash
A=$($PY -c "from paths import EXP; print(EXP)")/artifacts
for R in allpose8fas10 allpose9jbq10 allpose8av210 allpose8slb10; do
  $PY scripts/run_health.py --preflight --jobs $A/${R}_jobs.json \
      --out $A/sweep_out/$R --logs $A/${R}_logs --seeds 10 || { echo "PREFLIGHT FAILED $R"; break; }
done
```

Exit code 1 means **do not launch**. Then, one run at a time (each uses all 7 GPUs):

```bash
for R in allpose8fas10 allpose9jbq10 allpose8av210 allpose8slb10; do
  $PY scripts/sweep_sched.py --jobs $A/${R}_jobs.json --out $A/sweep_out/$R \
      --logs $A/${R}_logs --status $A/${R}_status.json --gpus 0,1,2,3,4,5,6 --n-sample 5
done
```

Attach a watcher for the run's life in another pane:
`$PY scripts/run_health.py --watch --jobs $A/${R}_jobs.json --out $A/sweep_out/$R --logs $A/${R}_logs --seeds 10 --interval 1800`

## 5. Only after ALL FOUR MSA-on arms finish — build and run the MSA-off twins

```bash
for R in allpose8fas10 allpose9jbq10 allpose8av210 allpose8slb10; do
  $PY scripts/build_nomsa_variant.py --run $R
done
```

This reuses the same 250 templates and mmCIF ids, changing only `pairedMsaPath` /
`unpairedMsaPath` to a 1-sequence a3m. **No new template ids are minted**, so the 4-char and
release-date traps cannot recur. Then preflight and run each `${R}nomsa` exactly as in step 4.

## 6. Verify — not optional

A condition failure is **invisible in the output**: an untemplated run, or one with the MSA still
attached, still emits 12,500 plausible poses. Three arms on this project were scored and believed for a
week while silently untemplated, and a second machine burned 50 minutes the same way last night. Per
run, in `$A/${R}_logs/`:

```bash
grep -h "Found 0 templates" $A/${R}_logs/*.log | wc -l     # MUST be 0
grep -ho "N_msa [0-9]*" $A/${R}_logs/*.log | sort -u        # MSA-on: large. MSA-off: N_msa 1
ls $A/sweep_out/$R/$R/*/seed_*/predictions/*_summary_confidence_sample_4.json | wc -l   # 2500
```

`Found 0 templates` usually means a template id isn't 4 characters, or the donor mmCIF is absent from
`$HOME/common/release_date_cache.json` — Protenix filters every hit through a release-date lookup
(`template_utils.py:369`) and silently skips date-unknown hits. The builder writes those entries, so if
you see this, report it rather than patching the cache.

**Report a FAILED run as failed. Do not score it and do not report its numbers.**

## 7. Send back

```bash
tar czf allpose_results.tgz -C $A/sweep_out allpose8fas10 allpose9jbq10 allpose8av210 allpose8slb10 \
    allpose8fas10nomsa allpose9jbq10nomsa allpose8av210nomsa allpose8slb10nomsa
```

Large — ~12,500 poses per run. If that's unwieldy, send the confidence JSONs and CIFs only, or tell me
and I'll set up scoring on your side. **Do not score there by default**: scoring needs OST and the
FoldBench ground truths, and both halves must go through one identical protocol
(`--fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq`, interface matched by chain name).

Also report per run: the three verification checks, wall-clock, and any run that failed or was left
incomplete.
