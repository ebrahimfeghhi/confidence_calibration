#!/usr/bin/env bash
# 800-pose ConforNet sweep: 20 independent runs x 4 save steps x 2 ConforNets x 5 samples.
#
# Each seed is a FULLY independent run -- fresh (phi1, phi2) from identity + new Gaussian noise, no
# shared weights or optimizer state. Trunk draws are seed*1000+step so no two runs overlap.
#
# Sequential on one GPU by design: k=2 backprop peaks at 69 GiB, so two runs will not co-reside on a
# 97 GiB card. To use more cards, run this with disjoint SEEDS on each.
#
# RESUMABLE: a seed whose training_loss.csv already has all steps is skipped, so re-running after an
# interruption costs nothing. Progress is appended per run, never buffered to the end.
set -uo pipefail

EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-4}
# Run seeds are spaced by STEPS so each run consumes a contiguous, disjoint block of the global seed
# line: run 1 uses 0-19, run 2 uses 20-39, ..., run 20 uses 380-399. The script itself just does
# seed+step, so this spacing IS the non-overlap guarantee -- change STEPS and this must change too.
STEPS=${STEPS:-20}
SEEDS=${SEEDS:-$(seq 0 "$STEPS" $(( 20 * STEPS - STEPS )))}
LOG=$EXP/ops/sweep_800.log

cd "$EXP" || exit 1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "sweep start: gpu=$GPU steps=$STEPS seeds=$(echo $SEEDS | tr '\n' ' ')"
done_n=0; fail_n=0; skip_n=0
for s in $SEEDS; do
  tag="8che_sweep_s${s}"
  csv="$EXP/artifacts/confornet/$tag/training_loss.csv"
  # complete = header + STEPS rows
  if [ -f "$csv" ] && [ "$(wc -l <"$csv")" -ge "$((STEPS + 1))" ]; then
    say "seed $s: already complete, skipping"; skip_n=$((skip_n+1)); continue
  fi
  say "seed $s: starting (tag=$tag)"
  $PY scripts/confornet_protenix.py --pdb 8che --gpu "$GPU" --k 2 \
      --seed "$s" --steps "$STEPS" --tag "$tag" >>"$EXP/ops/sweep_s${s}.log" 2>&1
  rc=$?
  n=$(find "$EXP/artifacts/confornet/$tag" -name "*.cif" 2>/dev/null | wc -l)
  if [ "$rc" -eq 0 ]; then
    say "seed $s: done rc=0, $n poses"; done_n=$((done_n+1))
  else
    say "seed $s: FAILED rc=$rc, $n poses -- see ops/sweep_s${s}.log; continuing"; fail_n=$((fail_n+1))
  fi
done

total=$(find "$EXP/artifacts/confornet" -path "*8che_sweep_s*" -name "*.cif" 2>/dev/null | wc -l)
say "sweep end: ok=$done_n failed=$fail_n skipped=$skip_n; TOTAL POSES=$total (target 800)"
