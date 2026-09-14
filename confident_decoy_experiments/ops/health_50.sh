#!/bin/bash
# Health watcher for the 50-seed arms (d50 then bo50), every 30 min -> ops/health_50.txt (latest)
# and ops/health_50_history.txt (appended).
#
# Flags the failure modes that have actually bitten this project, not just "is it alive":
#   - a worker session gone while arms remain
#   - a GPU idle while arms remain (the tail problem: fewer arms left than cards)
#   - an arm restarted repeatedly with 0 poses (the 8evd loop that burned a GPU for 16 h)
#   - CPU over the 64-core target
#   - another user appearing on one of our cards (OOM risk: no scheduler on this box)
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python; GPUS="0 1 2 3 5 6 7"; ME=$(id -un)
say(){ echo "[$(date +'%m-%d %H:%M:%S')] $*"; }
while true; do
  {
    # which stage is live
    stage=d50; qf=$A/d50_queue.txt; pre=d50w
    if [ -s "$A/bo50_queue.txt" ] && [ "$(tmux ls 2>/dev/null | grep -c '^bo50w')" -gt 0 ]; then
        stage=bo50; qf=$A/bo50_queue.txt; pre=bo50w
    fi
    # unif runs CONCURRENTLY on its own cards, so report it separately rather than as a stage
    uw=$(tmux ls 2>/dev/null | grep -c '^unifw' || true)
    ul=0
    while read -r r; do
        [ -z "$r" ] && continue
        n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$n" -lt 250 ] && ul=$((ul+1))
    done < "$A/unif_queue.txt"
    say "unif: arms_left=$ul workers=$uw"
    [ "$ul" -gt 0 ] && [ "$uw" -eq 0 ] && say "!! unif has no workers but $ul arms remain"
    left=0; inc=""
    while read -r r; do
      [ -z "$r" ] && continue
      n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
      if [ "$n" -lt 200 ]; then left=$((left+1)); inc="$inc $r:$n"; fi
    done < "$qf"
    workers=$(tmux ls 2>/dev/null | grep -c "^$pre" || true)
    say "stage=$stage  arms_left=$left  workers=$workers"

    [ "$left" -gt 0 ] && [ "$workers" -eq 0 ] && say "!! NO WORKERS but $left arms remain"

    for g in $GPUS; do
      u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g 2>/dev/null)
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
      # idle = no memory AND no util; brief 0% with memory held is just an arm boundary
      # Flag an idle card whenever ANY work remains anywhere -- d50, bo50 or unif. The old guard
      # required left >= workers, which is precisely backwards: idling happens at the TAIL, when
      # fewer arms remain than cards. That gap left four GPUs idle for hours on the d50 tail.
      tot=$((left + ul))
      if [ "${m:-0}" -lt 2000 ] && [ "${u:-0}" -eq 0 ] && [ "$tot" -gt 0 ]; then
        say "!! gpu$g IDLE (${m}MiB) while $tot arms remain (d50/bo50=$left unif=$ul)"
      fi
    done

    # foreign processes on our cards -- no scheduler here, so a co-tenant means both jobs OOM
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      o=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
      [ -n "$o" ] && [ "$o" != "$ME" ] && say "!! foreign process on a GPU: user=$o pid=$p"
    done

    # permanently-failing arm: many starts, still zero poses
    for r in $(echo "$inc" | tr ' ' '\n' | grep ':0$' | cut -d: -f1); do
      t=$(grep -h "START $r\b" $REPO/ops/${pre}*.log 2>/dev/null | wc -l)
      [ "$t" -gt 3 ] && say "!! $r started $t times, still 0 poses -- likely unrunnable"
    done

    c=$(ps -u $ME -o pcpu= | awk '{s+=$1} END {printf "%.0f", s/100}')
    [ "${c:-0}" -gt 64 ] && say "!! CPU ${c} cores (target 64)"
    e=$(grep -hcE "Traceback|CUDA out of memory" $REPO/ops/${pre}*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
    say "cores=${c}  errors=${e:-0}  disk=$(df -h /home | tail -1 | awk '{print $4}') free"
  } > $REPO/ops/health_50.txt 2>&1
  cat $REPO/ops/health_50.txt >> $REPO/ops/health_50_history.txt
  sleep 1800
done
