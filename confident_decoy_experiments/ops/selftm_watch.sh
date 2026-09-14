#!/bin/bash
# Watchdog + driver for the self-template pipeline. Every 10 min it:
#   1. reports arm progress for the live stage
#   2. BACKFILLS any idle card -- the tail (fewer arms than GPUs) is where cards have gone idle
#      every previous run, and a watcher that only reports is worth little
#   3. drives the wave-2 pipeline: stage1(29) -> build templates -> stage2(29)
# Writes ops/selftm_watch.txt (latest) and appends ops/selftm_watch_history.txt.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python; GPUS="0 1 2 3"; ME=$(id -un)
say(){ echo "[$(date +'%m-%d %H:%M:%S')] $*"; }

left_in(){ # $1=queue  $2=poses per arm -> count incomplete
    local n c=0
    while read -r r; do [ -z "$r" ] && continue
        n=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$n" -lt "$2" ] && c=$((c+1)); done < "$1"; echo "$c"
}
start_workers(){ # $1=session prefix  $2=worker script
    for g in $GPUS; do
        tmux has-session -t $1$g 2>/dev/null && continue
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null || echo 99999)
        [ "$mem" -ge 2000 ] && continue          # never start on top of another job
        tmux new-session -d -s $1$g -c $REPO
        tmux send-keys -t $1$g "bash $REPO/ops/$2 $g 2>&1 | tee -a $REPO/ops/$1$g.log" Enter
        say "  started $1$g on gpu$g"
    done
}

while true; do
  {
    # ---- stage 2 wave 1 -------------------------------------------------------------------
    w1=$(left_in $A/selftm_queue.txt 50)
    say "selftm wave1: $w1/100 arms left, workers=$(tmux ls 2>/dev/null | grep -c '^selftmw' || echo 0)"
    if [ "$w1" -gt 0 ]; then
        start_workers selftmw selftm_worker.sh
    else
        # ---- wave 2: stage 1 -------------------------------------------------------------
        s1b=$(left_in $A/s1b_queue.txt 5)
        if [ "$s1b" -gt 0 ]; then
            say "wave1 done -- stage1 wave2: $s1b/29 complexes left"
            [ -f "$REPO/ops/s1b_worker.sh" ] || sed 's#s1_queue.txt#s1b_queue.txt#; s#s1_locks#s1b_locks#' \
                $REPO/ops/s1_worker.sh > $REPO/ops/s1b_worker.sh
            start_workers s1bw s1b_worker.sh
        elif [ ! -f "$A/selftm2_queue.txt" ]; then
            say "stage1 wave2 done -- building self-templates (ids 8300+)"
            $PY $REPO/scripts/build_selfpose_tmpl2.py --run selftm2 --tid-prefix 8 \
                --tid-start 700 --seeds 10 2>&1 | tail -4
            [ -f "$REPO/ops/selftm2_worker.sh" ] || sed 's#selftm_queue.txt#selftm2_queue.txt#; s#selftm_locks#selftm2_locks#' \
                $REPO/ops/selftm_worker.sh > $REPO/ops/selftm2_worker.sh
        else
            w2=$(left_in $A/selftm2_queue.txt 50)
            say "selftm wave2: $w2/145 arms left"
            [ "$w2" -gt 0 ] && start_workers selftm2w selftm2_worker.sh || say "ALL DONE"
        fi
    fi

    # ---- health --------------------------------------------------------------------------
    for g in $GPUS; do
        u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
        [ "${m:-0}" -lt 2000 ] && [ "${u:-0}" -eq 0 ] && say "  !! gpu$g IDLE"
    done
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        o=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
        [ -n "$o" ] && [ "$o" != "$ME" ] && say "  !! foreign process on a GPU: $o"
    done
    # the silent failure that matters here: a run that used fewer than 5 templates
    # one template per arm now: "Found 0" means the id was rejected and the run is silently
    # untemplated, which is the failure that has bitten this project before.
    bad=$(grep -lh "Found 0 templates" $A/selftm*_logs/*.log 2>/dev/null | head -3 | tr '\n' ' ')
    [ -n "$bad" ] && say "  !! UNTEMPLATED ARM (Found 0): $bad"
    cores=$(ps -u $ME -o pcpu= | awk '{s+=$1} END {printf "%.0f", s/100}')
    say "cores=${cores} disk=$(df -h /home | tail -1 | awk '{print $4}')"
  } > $REPO/ops/selftm_watch.txt 2>&1
  cat $REPO/ops/selftm_watch.txt >> $REPO/ops/selftm_watch_history.txt
  sleep 600
done
