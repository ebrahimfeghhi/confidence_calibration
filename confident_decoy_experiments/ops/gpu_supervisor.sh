#!/usr/bin/env bash
# OVERNIGHT SUPERVISOR: keep GPUs 0/3/4 occupied at all times.
#
# The hard requirement is that a card is never observably free -- another user takes it within
# seconds (GPU 4 was lost that way once). So this does not merely chain jobs; every 20s it checks
# each card and, if nothing of ours is running there, starts the highest-priority remaining work.
#
# PRIORITY
#   1. pocket-constrained oracle run (ops/run_pocket.sh)   -- 25 interfaces, the experiment
#   2. deep-adaptive scaling         (ops/run_deep_adaptive.sh) -- 9 no-acceptable-pose complexes
# Each has its own atomic lock dir, so workers never collide and a restarted worker resumes cleanly.
#
# A claimer that finds nothing left EXITS, which is how priority 1 hands over to priority 2.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
LOG=ops/gpu_supervisor.log
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

pocket_left(){
  # Must count only interfaces that are incomplete AND UNCLAIMED. Counting merely-incomplete work
  # made an idle GPU start a claimer that immediately found everything taken and exited, so the
  # supervisor restarted it every 25s while the card sat at 0% -- a spin, not a remediation.
  for f in artifacts/pocket_cd_jsons/*_pocket.json; do
    [ -e "$f" ] || continue
    t=$(basename "$f" _pocket.json)
    [ -f "artifacts/pocketcd_locks/$t.dead" ] && continue
    [ -d "artifacts/pocketcd_locks/$t" ] && continue          # already claimed by another GPU
    n=$(find "artifacts/sweep_out/pocketcd/pocketcd_$t" -name '*_sample_*.cif' 2>/dev/null | wc -l)
    [ "$n" -lt 250 ] && { echo 1; return; }
  done
  echo 0
}
gpu_busy(){
  for p in $(pgrep -f "runner/inference.py" 2>/dev/null); do
    [ -r /proc/$p/environ ] || continue
    g=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ "$g" = "$1" ] && { echo 1; return; }
  done
  # a claimer between complexes still counts as busy
  tmux has-session -t "cd_$1" 2>/dev/null && { echo 1; return; }
  tmux has-session -t "da_$1" 2>/dev/null && { echo 1; return; }
  echo 0
}
say "supervisor started, watching GPUs 0 3 4"
idle_count=0
while true; do
  for g in 0 3 4; do
    if [ "$(gpu_busy $g)" = "1" ]; then continue; fi
    if [ "$(pocket_left)" = "1" ]; then
      say "GPU$g idle -> starting POCKET claimer"
      tmux kill-session -t "cd_$g" 2>/dev/null
      tmux new-session -d -s "cd_$g" "cd $REPO && ARM=pocketcd JD=$REPO/artifacts/pocket_cd_jsons bash ops/run_pocket.sh $g 2>&1 | tee -a ops/pocketcd_g${g}.log"
    else
      say "GPU$g idle, pocket complete -> starting DEEP-ADAPTIVE claimer"
      tmux kill-session -t "da_$g" 2>/dev/null
      tmux new-session -d -s "da_$g" "cd $REPO && bash ops/run_deep_adaptive.sh $g 2>&1 | tee -a ops/deepadapt_g${g}.log"
    fi
    sleep 5
  done
  sleep 20
done
