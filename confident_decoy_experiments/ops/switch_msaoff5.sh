#!/usr/bin/env bash
# Hand each GPU from the wave-1 claimer (38-complex list) to a wave-2 claimer (full 49) as it drains.
#
# WHY NOT JUST RESTART: the wave-1 claimers are executing an older inode of run_msaoff5.sh (Write
# replaced the file, so they keep reading the unlinked original) and will never see the 49 list. But
# they exit by themselves once nothing in the 38 list is claimable, and the 49 list is ascending by
# token count, so the 11 extra complexes -- the LARGEST, 1192-2361 tokens -- were going to run last no
# matter what. Waiting therefore costs nothing and avoids killing a worker mid-seed.
#
# SAFE TO OVERLAP: wave-2 claimers coordinate with any still-running wave-1 claimer through the SAME
# mkdir lock dir, so a complex can never be taken twice. The only thing to avoid is two workers on one
# GPU (VRAM contention), which is why each GPU is handed over only after ITS OWN wave-1 session ends.
#
# Nothing here kills anything: the wave-1 worker holds its card until the instant its replacement
# starts, so the GPU is never observably free for someone else to claim.
set -o pipefail

REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
GPUS=${GPUS:-"0 3 4"}
POLL=${POLL:-10}
LOG=$REPO/ops/switch_msaoff5.log
cd "$REPO" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "wave-2 (49-complex) handover watcher armed for GPUs: $GPUS (poll ${POLL}s)"
declare -A flipped
while true; do
  pending=0
  for g in $GPUS; do
    if [ "${flipped[$g]}" = 1 ]; then
      tmux has-session -t "msa49_$g" 2>/dev/null && pending=1
      continue
    fi
    if tmux has-session -t "msaclaim$g" 2>/dev/null; then pending=1; continue; fi   # wave-1 still up
    say "GPU$g wave-1 drained -> starting wave-2 claimer (49 list)"
    tmux new-session -d -s "msa49_$g" "cd $REPO && bash ops/run_msaoff5.sh $g 2>&1 | tee -a ops/msa49_g${g}.log"
    sleep 3
    if tmux has-session -t "msa49_$g" 2>/dev/null; then
      say "GPU$g wave-2 up"; flipped[$g]=1; pending=1
    else
      say "GPU$g wave-2 FAILED to start -- retry next cycle"; pending=1
    fi
  done
  [ "$pending" = 0 ] && { say "all GPUs handed over and all wave-2 claimers exited -- 49-complex sweep complete"; break; }
  sleep "$POLL"
done
