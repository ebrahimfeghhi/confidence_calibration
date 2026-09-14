#!/usr/bin/env bash
# Keep the 250-donor run on up to THREE GPUs, taking cards only as they genuinely free up.
#
# HARD CAP OF 3 here (the standing rule is a maximum of 4 overall, and the user asked for 3 for this
# run). It never exceeds that, and it never touches a card another user is on.
#
# A GPU counts as AVAILABLE only if it has >= 60 GiB free AND no process from another user. Protenix
# needs 25-45 GiB at 1474 tokens; taking a card with less means an immediate CUDA OOM, which is what
# consumed 36 donors earlier today.
#
# EACH WORKER GETS A HOLDER. scripts/gpu_hold.py keeps ~10 GiB resident for the life of the run, so
# the card shows an ebrahim process even during our CPU phases -- template building and PXMeter
# scoring leave the GPU at 0 MiB / 0% for ~30% of the wall clock, which is exactly how a neighbour
# came to launch 73.5 GiB jobs onto cards we were using. The holder is a signal, not a reservation:
# 10 of 97 GiB blocks nobody, and reserving our real working set would make our own inference OOM.
#
# It also releases claims stranded by dead workers -- an orphaned claim is never retried by anyone.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
M=$REPO/artifacts/allpose20
LOG=$M/scale.log
MAXG=${MAXG:-3}
CAND="2 0 1 6 5"          # never 3/4/7 -- shawn; order puts the currently-free card first
cd "$REPO" || exit 1

while true; do
  ts=$(date '+%F %H:%M:%S')
  done_n=$(ls "$M/.done" 2>/dev/null | wc -l)
  rows=$(( $(wc -l < "$M/results.csv" 2>/dev/null || echo 1) - 1 ))

  # release claims whose owning worker is gone
  for c in $(ls "$M/.claims" 2>/dev/null); do
    [ -e "$M/.done/$c" ] && continue
    own=$(cat "$M/.claims/$c/owner" 2>/dev/null)
    [ -n "$own" ] || continue
    tmux has-session -t "apw$own" 2>/dev/null || { rm -rf "$M/.claims/$c"; \
      echo "[$ts] released stranded claim $c (GPU$own gone)" >> "$LOG"; }
  done

  active=$(tmux ls 2>/dev/null | grep -cE '^apw[0-9]:')
  if [ "$active" -lt "$MAXG" ]; then
    for g in $CAND; do
      [ "$active" -ge "$MAXG" ] && break
      tmux has-session -t "apw$g" 2>/dev/null && continue
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ -n "$free" ] || continue
      [ "$free" -lt 60000 ] && continue
      # never share a card with another user
      other=0
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
        u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
        [ -n "$u" ] && [ "$u" != "ebrahim" ] && other=1
      done
      [ "$other" = "1" ] && continue
      tmux has-session -t "hold$g" 2>/dev/null || tmux new-session -d -s "hold$g" -c "$REPO" \
        "CUDA_VISIBLE_DEVICES=$g /home/ebrahim/Protenix/.venv/bin/python scripts/gpu_hold.py --gib 10 --stop $M/HOLD_STOP"
      sleep 5
      tmux new-session -d -s "apw$g" -c "$REPO" "GPU=$g NS=20 bash ops/allpose20_worker2.sh"
      active=$((active+1))
      echo "[$ts] took GPU$g (${free}MiB free) -> worker + holder; now $active/$MAXG" >> "$LOG"
    done
  fi

  echo "[$ts] donors $done_n/250  rows $rows  workers $(tmux ls 2>/dev/null | grep -cE '^apw[0-9]:')/$MAXG  holders $(tmux ls 2>/dev/null | grep -cE '^hold[0-9]:')" >> "$LOG"
  [ "$done_n" -ge 250 ] && { echo "[$ts] ALL DONORS DONE" >> "$LOG"; touch "$M/HOLD_STOP"; break; }
  sleep 120
done
