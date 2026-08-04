# Confident-decoy experiments — backup

Working directory `/home/ebrahim/confident_decoy_experiments/` is **47 GB and not a git repo**.
This mirror holds everything small enough to version, so a dead server costs GPU time but not results.

## Backed up here (35 MB)

| path | what |
|---|---|
| `scripts/` | all analysis + pipeline code (82 files) |
| `configs/` | per-run configs, 2026-07-28 deployable-selection sweep |
| `artifacts/*.json` | **the scored results** — every headline number is re-tallyable from these |
| `artifacts/scores/` | 2026-07-28 sweep scores (topdockq, topiptm, oracle, abonly, sposefv …) |
| `artifacts/provenance/` | repo commits, checkpoint hash, tool versions, script sha256 per run |
| `artifacts/{native,sweep,pddag,relabel}_{tmpl,a3m}/` | template definitions — which pose donated which fold |
| `artifacts/logs/` | run logs, for diagnosing past failures |
| `FINDINGS.md`, `RUNS.md`, `PROJECT_README.md` | project docs from `/home/ebrahim/` |

## NOT backed up (regenerable, but expensive)

- `artifacts/*_ost/` — OST score caches (~30 GB). Recomputable on CPU from predictions + ground truths.
- `artifacts/sweep_out/`, `native_run_out/`, `pae_out/` — prediction CIFs. **GPU-expensive**: days of
  compute across the template arms.
- `/home/ebrahim/Protenix/foldbench_abag/predictions_template/` — the free-run pose pools, the
  foundation of every analysis. Also GPU-expensive.
- `/home/ebrahim/mmcif/*.cif` — registered fake-old template structures. Rebuildable by the builder
  scripts, but only if the source prediction CIFs still exist.

**If the box died right now:** every conclusion in the report survives, because the scored JSONs are
here. What is lost is the ability to re-derive new numbers from the poses without re-running Protenix.

Re-sync:
```bash
D=experiments/confident_decoy SRC=/home/ebrahim/confident_decoy_experiments
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' $SRC/scripts/ $D/scripts/
rsync -a --include='*.json' --exclude='*' $SRC/artifacts/ $D/artifacts/
```
