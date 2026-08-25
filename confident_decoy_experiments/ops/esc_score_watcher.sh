#!/bin/bash
# Scores finished escalation poses every 30 min. Read-only w.r.t. the GPU runs: it only reads CIFs
# and writes OST caches + artifacts/esc_status.json, so it can never disturb generation.
# Serialised against itself so two scoring passes never overlap on CPU.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=$REPO/ops/esc_score_watcher.log
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }
say "escalation score watcher started (every 30 min)"
while true; do
    if pgrep -u "$(id -un)" -f "python.*scripts/score_esc\.py" >/dev/null; then
        say "previous scoring pass still running -- skipping this tick"
    else
        say "scoring pass starting"
        $PY "$REPO/scripts/score_esc.py" > "$REPO/ops/esc_score_latest.txt" 2>&1
        tail -1 "$REPO/ops/esc_score_latest.txt" | tee -a "$LOG"
    fi
    sleep 1800
done
