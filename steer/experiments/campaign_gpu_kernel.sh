#!/usr/bin/env bash
# campaign_gpu_kernel.sh <gpu> <struct1> [struct2] [struct3] ...
#
# Same recipe as campaign_gpu_jointz0.sh (K=3/M=5 label-free repulsion, joint-z0
# on, targeted AB×C interface loss -- no --intra-c, no --full-disto), but with
# --loss kernel: the summed Gaussian-kernel basin-filling repulsion instead of
# the plain mean-over-anchors loss (see docs/summed_kernel_loss.md). Bandwidth
# auto-calibrates from the anchor-to-anchor NN spread unless KERNELH is set.
# Runs targets sequentially on ONE gpu; each stops on a DockQ>=0.23 hit or the
# time budget, then the next target starts. Tag suffix "_kernel" keeps outputs
# separate from any mean-loss run of the same target.
#
#   Usage:  ./campaign_gpu_kernel.sh 0 8jyr 8urf
#
# Knobs via env: BUDGET (12h=43200), NS, STEPS, LB, RUNS, CYC, CS, LRZ0 (0.01),
# KERNELH (0=auto), KERNELC (1.0), INTRAC_TARGET (target name that also gets
# --intra-c, e.g. INTRAC_TARGET=8vvl for the one structure that misfolded its
# antigen -- see docs/z0_loss_scope_results.md / joint-z0-intra-antigen-loss).
set -u
PY=/home/ebrahim/esm/.venv/bin/python
EXP=/home/ebrahim/tts_abag/steer/experiments
GPU=${1:?need gpu index}; shift
TARGETS=("$@")
[ ${#TARGETS[@]} -ge 1 ] || { echo "need at least one target"; exit 1; }
BUDGET=${BUDGET:-43200}
NS=${NS:-5}; STEPS=${STEPS:-200}; LB=${LB:-2}; RUNS=${RUNS:-512}; CYC=${CYC:-16}; CS=${CS:-32}
LRZ0=${LRZ0:-0.01}
KERNELH=${KERNELH:-0}
KERNELC=${KERNELC:-1.0}
INTRAC_TARGET=${INTRAC_TARGET:-}

run_campaign () {
  local s=$1 extra="" tag="${1}_k3m5_kernel"
  if [ "$s" = "$INTRAC_TARGET" ]; then extra="--intra-c"; tag="${1}_k3m5_kernel_intrac"; fi
  echo "=== [$(date '+%F %T')] GPU $GPU: campaign $s (tag $tag) START  budget=${BUDGET}s loss=kernel(h=${KERNELH},c=${KERNELC}) joint-z0 lr-z0=${LRZ0} extra='${extra}' ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$EXP/repel_proven.py" \
      --target "$s" --lever slm --M 5 --K 3 --lr 0.3 --init-frac 0.05 \
      --loss kernel --kernel-h "$KERNELH" --kernel-c "$KERNELC" \
      --joint-z0 --lr-z0 "$LRZ0" $extra \
      --trials 1 --runs "$RUNS" --no-score --fused --chunk-size "$CS" \
      --cycle-decode "$CYC" --stop-dockq 0.23 --time-budget "$BUDGET" \
      --n-samples "$NS" --n-steps "$STEPS" --launch-batch "$LB" --tag "$tag" \
      2>&1 | tee "$EXP/log_${tag}.txt"
  echo "=== [$(date '+%F %T')] GPU $GPU: campaign $s DONE ==="
}

for s in "${TARGETS[@]}"; do
  run_campaign "$s"
done
echo "=== [$(date '+%F %T')] GPU $GPU: ALL campaigns finished (${TARGETS[*]}) ==="
