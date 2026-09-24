#!/usr/bin/env bash
# Babysit a sweep: report progress, restart dead workers, run the scorers. One cycle per --interval.
#
#   watch.sh --exp <dir> --gpus 1,5,6 [--templates <dir>] [--interval 1800] <reporter.py> [...]
#
# A watcher that only prints is useless unattended -- by morning it has logged the same failure sixteen
# times and nothing has run. So it restarts what it can and prefixes ESCALATE on what it cannot, so one
# grep finds it. Restarting also recovers the dead worker's claimed units, because folder.py releases
# claims that have no results when it starts.
#
# Worker launching is DELEGATED to call_folder.py. Two scripts each building their own `tmux
# new-session` invented two naming schemes for the same session -- the index (w0/w1/w2) here, the GPU
# id (w1/w5/w6) there -- and the liveness check knew one of them, so launching the other way would
# declare live workers dead and start a second set on cards that already had one. One launcher, one
# convention: sessions are <experiment>w<gpu>.
#
# EACH CYCLE
#   1. count claims belonging to THIS queue, and restart any worker whose session is gone
#   2. print workers / claims / rows
#   3. grep worker logs for tracebacks and OOM -> ESCALATE
#   4. print GPU memory and utilisation for the cards in use
#   5. run each reporter, writing the combined output to <exp>/table.txt
set -u

EXP=""; GPUS=""; TMPL=""; INT=1800; REPORTERS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --exp)       EXP=$2; shift 2 ;;
    --gpus)      GPUS=$2; shift 2 ;;
    --templates) TMPL=$2; shift 2 ;;
    --interval)  INT=$2; shift 2 ;;
    *)           REPORTERS+=("$1"); shift ;;
  esac
done
[ -n "$EXP" ]  || { echo "usage: watch.sh --exp <dir> --gpus 1,5,6 [--templates <dir>] <reporter.py>"; exit 1; }
[ -n "$GPUS" ] || { echo "--gpus is required: this machine is shared and devices are never guessed"; exit 1; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # reporters and call_folder.py live beside this
PY=/home/ebrahim/Protenix/.venv/bin/python
NAME=$(basename "$EXP")
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

while true; do
  echo "================ $(date '+%F %H:%M')  [$NAME] ================"

  # Claims belonging to THIS queue only. A run that reuses folds from an earlier sweep pre-creates a
  # claim per reused donor, so a raw `ls claims | wc -l` exceeds the queue length, the restart branch
  # below becomes unreachable, and a dead worker idles its GPU all night.
  total=$(wc -l < "$EXP/queue.tsv")
  qtags=$(mktemp); ctags=$(mktemp)
  awk -F'\t' 'NF>=6{print $2"_"$3"_"$4"_"$5}' "$EXP/queue.tsv" | sort -u > "$qtags"
  ls "$EXP/claims" 2>/dev/null | sort -u > "$ctags"
  claimed=$(comm -12 "$qtags" "$ctags" | wc -l)
  carried=$(comm -13 "$qtags" "$ctags" | wc -l)
  rm -f "$qtags" "$ctags"

  if [ "$claimed" -lt "$total" ]; then
    for g in "${GPU_LIST[@]}"; do
      if ! tmux has-session -t "${NAME}w${g}" 2>/dev/null; then
        echo "REMEDIATE: worker on GPU $g gone -- restarting"
        (cd "$HERE" && $PY call_folder.py --queue "$EXP/queue.tsv" --out "$EXP" \
            ${TMPL:+--templates "$TMPL"} --gpus "$g" --run >/dev/null)
      fi
    done
  fi

  echo "workers=$(pgrep -u "$USER" -f "$NAME/queue.tsv" | wc -l)  claimed=$claimed/$total (+$carried carried)  rows=$(( $(wc -l < "$EXP/results.csv" 2>/dev/null || echo 1) - 1 ))"

  for g in "${GPU_LIST[@]}"; do
    # grep -c prints 0 AND exits 1 when nothing matches, so `|| echo 0` would yield "0\n0" and break
    # the numeric test below; `; true` keeps the count and swallows the exit status
    n=$(grep -cE "Traceback|OutOfMemory|CUDA out of memory" "$EXP/w${g}.log" 2>/dev/null; true)
    [ "${n:-0}" -gt 0 ] && echo "ESCALATE: GPU $g worker has $n error lines (grep -nE 'Traceback|out of memory' $EXP/w${g}.log)"
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$GPUS" \
    | sed 's/^/  gpu /'

  : > "$EXP/table.txt"
  for r in "${REPORTERS[@]}"; do (cd "$HERE" && $PY "$r" >> "$EXP/table.txt" 2>&1); done
  cat "$EXP/table.txt"

  sleep "$INT"
done
