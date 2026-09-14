#!/usr/bin/env bash
# Wait for hybrid25b (batch B) to drain, then generate 25 MORE free seeds for every PXM-2025 complex.
#
# WHY QUEUE: batch B still holds all three GPUs. Starting now would put two inference jobs on each
# card; hybrid25b_worker.sh waits for 45 GB free, so the new workers would spin and the old ones would
# slow down.
#
# THE TRIGGER is the absence of hybrid25b_worker processes, not a unit count -- a worker exits only
# when no unclaimed units remain, which is the real definition of drained. Counting .done can fire
# while the last units are still in flight, and never fires at all if one dies and is released for a
# retry nobody picks up.
#
# SCOPE: seeds 300075-300099 for all 131 complexes, free (no self-templating), into the existing
# artifacts/pxm2025/out tree so scoring sees one pool of 75 free seeds per complex. ~104 GPU-hours by
# the cost model, so roughly 35 h across three cards.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
GPUS=${GPUS:-"2 5 6"}
SEED_LO=${SEED_LO:-300075}
NSEEDS=${NSEEDS:-25}
cd "$REPO" || exit 1
LOG=ops/pxm_more_queue.log
say () { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

say "queued: waiting for hybrid25b to drain, then seeds $SEED_LO-$((SEED_LO+NSEEDS-1)) x 131 complexes"
while pgrep -u "$(id -u)" -f 'hybrid25b_worker.sh' >/dev/null 2>&1; do sleep 120; done
say "batch B workers exited; $(ls artifacts/hybrid25b/.done 2>/dev/null | wc -l)/1125 units done"

if pgrep -u "$(id -u)" -f 'run_pxm2025_more.sh' >/dev/null 2>&1; then
  say "ABORT: the second free sweep is already running"; exit 1
fi
sleep 60                                   # let in-flight CUDA memory actually release

for g in $GPUS; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null)
  say "GPU$g free ${free}MiB -> launching free-seed worker"
  if tmux has-session -t "pxm$g" 2>/dev/null; then
    tmux send-keys -t "pxm$g" "cd $REPO && GPU=$g SEED_LO=$SEED_LO NSEEDS=$NSEEDS bash ops/run_pxm2025_more.sh" C-m
  else
    tmux new-session -d -s "pxm$g" "cd $REPO && GPU=$g SEED_LO=$SEED_LO NSEEDS=$NSEEDS bash ops/run_pxm2025_more.sh"
  fi
  sleep 10
done

sleep 60
n=$(pgrep -u "$(id -u)" -fc 'run_pxm2025_more.sh' 2>/dev/null || echo 0)
say "free-seed workers running: $n (expected $(echo $GPUS | wc -w))"
[ "$n" -ge 1 ] || say "!! DID NOT START -- by hand: GPU=<n> SEED_LO=$SEED_LO NSEEDS=$NSEEDS bash ops/run_pxm2025_more.sh"
