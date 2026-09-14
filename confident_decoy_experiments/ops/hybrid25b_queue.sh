#!/usr/bin/env bash
# Wait for the hybrid25 sweep to drain, then start batch B (decoy complexes, donor seeds 300025-49).
#
# WHY A QUEUE RATHER THAN LAUNCHING NOW: the three GPUs are still finishing hybrid25. Starting batch B
# alongside it would put two inference jobs on each card and OOM both -- hybrid25_worker.sh already
# waits for 45 GB free, so the new workers would simply spin, and the old ones would slow down.
#
# THE TRIGGER is the absence of hybrid25_worker processes, not a unit count. A worker exits only when
# it finds no unclaimed units left, which is the real definition of "drained"; counting .done can fire
# early while the last three units are still in flight, and can never fire at all if a unit dies and
# is released for a retry that no longer has a worker to pick it up.
#
# SAFETY: refuses to start if batch B's own workers are already running (double-launch after a manual
# start), and re-checks that the GPUs are actually free before each launch.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
GPUS=${GPUS:-"2 5 6"}
NS=${NS:-20}
cd "$REPO" || exit 1
LOG=ops/hybrid25b_queue.log

say () { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

say "queued: waiting for hybrid25 to drain (batch B = $(( $(wc -l < artifacts/hybrid25b/units.csv) - 1 )) units)"

while pgrep -f 'hybrid25_worker.sh' >/dev/null 2>&1; do sleep 120; done
say "hybrid25 workers have exited; $(ls artifacts/hybrid25/.done 2>/dev/null | wc -l)/3075 units done"

if pgrep -f 'hybrid25b_worker.sh' >/dev/null 2>&1; then
  say "ABORT: batch B workers already running"; exit 1
fi

# let any in-flight CUDA memory actually release before we claim the cards
sleep 60

for g in $GPUS; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null)
  say "GPU$g free ${free}MiB -> launching batch B worker (NS=$NS)"
  if tmux has-session -t "hyw$g" 2>/dev/null; then
    tmux send-keys -t "hyw$g" "cd $REPO && GPU=$g NS=$NS bash ops/hybrid25b_worker.sh" C-m
  else
    tmux new-session -d -s "hyw$g" "cd $REPO && GPU=$g NS=$NS bash ops/hybrid25b_worker.sh"
  fi
  sleep 10
done

sleep 60
n=$(pgrep -fc 'hybrid25b_worker.sh' 2>/dev/null || echo 0)
say "batch B workers running: $n (expected >= $(echo $GPUS | wc -w))"
[ "$n" -ge 1 ] || say "!! BATCH B DID NOT START -- run by hand: GPU=<n> NS=$NS bash ops/hybrid25b_worker.sh"
