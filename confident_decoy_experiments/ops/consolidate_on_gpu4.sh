#!/usr/bin/env bash
# Watch for shawn releasing GPU 4, then collapse the two split drivers into ONE that balances all
# three GPUs over the whole remaining complex list.
#
# The split exists only because GPU 4 is shared: small complexes go there (they fit in shawn's
# leftover memory) and large ones stay on 3+5. Once GPU 4 is ours outright that constraint is gone and
# a single scheduler keeps all three cards busy with whatever is next, which is both simpler and
# better balanced.
#
# KILLS ONLY THE DRIVERS, NEVER THE WORKERS. Workers hold the CUDA contexts; killing them frees the
# cards and on this box a released GPU is claimed by someone else within seconds -- that is exactly how
# GPU 4 was lost earlier today. The new driver picks each card up as its worker finishes.
#
# Requires GPU 4 to be shawn-free on TWO consecutive checks, so a momentary gap between his cells does
# not trigger a needless restart.
set -o pipefail

EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
LOG=$EXP/ops/consolidate.log
cd "$EXP" || exit 1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

others_on_gpu4() {
  local u4
  u4=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 4 | tr -d ' ')
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$u4" | cut -d, -f2 | tr -d ' ' \
    | while read -r p; do ps -o user= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u | grep -v '^ebrahim$'
}

say "watching GPU 4; will consolidate to one 3-GPU driver when it is ours alone"
clear_count=0
while true; do
  if ! pgrep -f "bash ops/sweep_selftmpl.sh" >/dev/null; then
    say "no drivers running -- sweep already finished or stopped; exiting"; exit 0
  fi
  if [ -z "$(others_on_gpu4)" ]; then
    clear_count=$((clear_count + 1))
  else
    clear_count=0
  fi

  if [ "$clear_count" -ge 2 ]; then
    say "GPU 4 is free of other users -- consolidating"
    for d in $(pgrep -f "bash ops/sweep_selftmpl.sh"); do
      kill "$d" 2>/dev/null && say "  stopped driver $d (workers left running, cards stay held)"
    done
    sleep 3
    held=$(pgrep -fa "gen_selftmpl.py --pdb" | grep -oP -- "--gpu \K[0-9]+" | sort -u | tr '\n' ' ')
    say "  workers still holding GPUs: ${held:-none}"
    tmux has-session -t stall3 2>/dev/null || tmux new-session -d -s stall3 -c "$EXP"
    tmux send-keys -t stall3 \
      'MAX_SLOTS=3 EXCLUDE_GPUS=0,1,2,6,7 LIST=ops/sweep_49_order.txt bash ops/sweep_selftmpl.sh' C-m
    sleep 30
    d=$(pgrep -f "bash ops/sweep_selftmpl.sh" | head -1)
    say "  new driver $d: 3 GPUs, full 49-complex list (completed ones are skipped)"
    exit 0
  fi
  sleep 60
done
