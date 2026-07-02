#!/usr/bin/env bash
# campaign_8vvl_intrac.sh <gpu>
#
# 8vvl-specific comparison: intra-antigen (C×C) block added to the proven AB×C loss,
# run 1 with --joint-z0 (z0+s_lm), run 2 s_lm-only. Same recipe/budget otherwise as
# campaign_gpu.sh. 8vvl's antigen was confirmed misfolded (GT lDDT ~0.2, pLDDT ~30
# in decoded structures) -- this tests whether making the loss "see" the antigen's
# own fold (not just the interface) can fix it.
set -u
PY=/home/ebrahim/esm/.venv/bin/python
EXP=/home/ebrahim/tts_abag/steer/experiments
GPU=${1:?need gpu index}
BUDGET=${BUDGET:-43200}
NS=${NS:-5}; STEPS=${STEPS:-200}; LB=${LB:-2}; RUNS=${RUNS:-512}; CYC=${CYC:-16}; CS=${CS:-32}
LRZ0=${LRZ0:-0.01}

run () {
  local tag=$1; shift
  echo "=== [$(date '+%F %T')] GPU $GPU: 8vvl intra-c run (tag $tag) START  budget=${BUDGET}s  args: $* ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$EXP/repel_proven.py" \
      --target 8vvl --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 --loss mean \
      --intra-c "$@" \
      --trials 1 --runs "$RUNS" --no-score --fused --chunk-size "$CS" \
      --cycle-decode "$CYC" --stop-dockq 0.23 --time-budget "$BUDGET" \
      --n-samples "$NS" --n-steps "$STEPS" --launch-batch "$LB" --tag "$tag" \
      2>&1 | tee "$EXP/log_${tag}.txt"
  echo "=== [$(date '+%F %T')] GPU $GPU: 8vvl intra-c run (tag $tag) DONE ==="
}

run 8vvl_intrac_jz0 --joint-z0 --lr-z0 "$LRZ0"
run 8vvl_intrac_slmonly
echo "=== [$(date '+%F %T')] GPU $GPU: BOTH 8vvl intra-c runs finished ==="
