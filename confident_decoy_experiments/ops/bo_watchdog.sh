#!/bin/bash
# Auto-remediation for hot-looping arms. Every 10 min.
#
# THE FAILURE THIS EXISTS FOR. wnm_worker.sh claims any arm whose pose count is below target, runs it,
# releases the lock, and loops. If an arm can NEVER produce poses -- e.g. 8evd, 2718 tokens against
# protenix-v2's 2560 hard assert -- it stays permanently below target, so the same worker re-claims it
# forever and that GPU does no useful work. 8evd burned GPU1 for ~16 h and 464 restarts before anyone
# looked. The health watcher printed the fault the whole time; nothing acted on it.
#
# RULE: an arm STARTed more than MAX_TRIES times while still holding 0 poses is not transient. It is
# dropped from the queue and recorded in bo_dropped.txt with its last error, so the count of runnable
# complexes never silently disagrees with the design without an explanation on disk.
#
# Deliberately conservative: only 0-pose arms are ever dropped. An arm producing poses slowly, or
# partially, is left alone -- restarts are normal there and dropping it would lose real work.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
MAX_TRIES=3
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$REPO/ops/bo_watchdog.log"; }
say "watchdog started (drop 0-pose arms after $MAX_TRIES starts; every 10 min)"
while true; do
    dropped=0
    while read -r r; do
        [ -z "$r" ] && continue
        poses=$(find "$A/sweep_out/$r" -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$poses" -gt 0 ] && continue
        tries=$(grep -h "START $r\b" $REPO/ops/bow*.log 2>/dev/null | wc -l)
        [ "$tries" -le "$MAX_TRIES" ] && continue
        err=$(grep -hoE "AssertionError:.*|CUDA out of memory[^,]*" \
              "$A/${r}_logs"/*.log 2>/dev/null | tail -1)
        say "!! DROPPING $r -- $tries starts, 0 poses. last error: ${err:-unknown}"
        grep -v "^${r}$" "$A/bo_queue.txt" > "$A/wnm_queue.tmp" && mv "$A/wnm_queue.tmp" "$A/wnm_queue.txt"
        echo "$r | $tries starts | ${err:-unknown}" >> "$A/bo_dropped.txt"
        rm -rf "$A/bo_locks/$r"
        dropped=$((dropped+1))
    done < "$A/bo_queue.txt"
    n=$(wc -l < "$A/bo_queue.txt")
    if [ "$dropped" -gt 0 ]; then say "dropped $dropped arm(s); $n remain in queue"; fi
    sleep 600
done
