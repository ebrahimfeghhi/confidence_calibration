#!/usr/bin/env bash
# Babysit a sweep: report progress, restart dead workers, run the scorers.
#
#   watch.sh --exp <dir> --gpus 1,5,6 [--templates <dir>] [--interval 1800] <reporter.py> [...]
#
# SUMMARY
#   Runs one cycle per interval so a sweep can be left unattended. Worker launching is delegated to
#   call_folder.py, so session naming has one owner: <experiment>w<gpu>.
#
#   Each cycle: restart any worker whose tmux session is gone, print workers/claims/rows, grep the
#   worker logs for tracebacks and OOM, print GPU state, run each reporter.
#
# INPUTS
#   --exp        the experiment directory: queue.tsv, claims/, results.csv, w<gpu>.log
#   --gpus       which devices the sweep uses; drives both the restart loop and the GPU report
#   --templates  passed through to call_folder.py when restarting a templated run
#   --interval   seconds between cycles
#   reporters    python scripts run from this directory, output collected in order
#
# OUTPUTS
#   stdout        the cycle report; ESCALATE prefixes anything it cannot fix itself
#   <exp>/table.txt   the reporters' combined output, overwritten each cycle
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

  # claims matching THIS queue only: a reused-fold run pre-creates claims, so a raw count of the
  # directory exceeds the queue length and the restart branch below never fires
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
    # grep -c exits 1 when nothing matches; `; true` keeps the count without failing the assignment
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
