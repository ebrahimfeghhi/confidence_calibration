#!/usr/bin/env bash
# Monitor seeds 0-11 across 4 GPUs; launch seeds 12-23 when all finish.
set -euo pipefail

# --- resolve repo root and load centralized paths from paths.py ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ ! -f "$REPO_ROOT/paths.py" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
eval "$(python3 "$REPO_ROOT/paths.py" --sh)"

LOG_DIR="$ESM_PREDICTIONS/esmfold2fast_tts/logs"
GPUS=(0 1 2 3)
NEXT_SEEDS=(12 13 14 15 16 17 18 19 20 21 22 23)
COMPLEXES=(9au1 8urf 8w90 8bf4 8hxq 8zp2 7yxu 8aon 8g4p 8gye 8be1 8byu 8pmz 8rp8 8jeo 8j7t 8en5 8tuz 7x8p 8fbw)
CHECK_INTERVAL=1200  # 20 minutes

all_done() {
    for gpu in "${GPUS[@]}"; do
        local log="$LOG_DIR/gpu${gpu}.log"
        if [[ ! -f "$log" ]]; then return 1; fi
        if ! tail -3 "$log" | grep -q "^All done\.\|^Done\."; then return 1; fi
    done
    return 0
}

progress_summary() {
    for gpu in "${GPUS[@]}"; do
        local log="$LOG_DIR/gpu${gpu}.log"
        if [[ ! -f "$log" ]]; then
            echo "  GPU $gpu: log not found"
            continue
        fi
        local last_struct
        last_struct=$(grep -oP '^\s+\[\K\d+(?=/20\] )' "$log" | tail -1 || echo "?")
        local current
        current=$(grep -oP '(?<=\] )\w+$' "$log" | tail -1 || echo "?")
        local done_flag=""
        tail -3 "$log" | grep -q "^All done\.\|^Done\." && done_flag=" [DONE]"
        echo "  GPU $gpu: structure ${last_struct}/20 (${current})${done_flag}"
    done
}

echo "$(date): Monitor started. Checking every $((CHECK_INTERVAL/60)) min."
echo "Waiting for seeds 0-11 to complete on all 4 GPUs..."
echo ""

while true; do
    echo "$(date): Progress check"
    progress_summary
    echo ""

    if all_done; then
        echo "$(date): All 4 GPUs finished seeds 0-11. Launching seeds 12-23..."
        "$ESM_PYTHON" "$SCRIPT_DIR/run_foldbench_abag_parallel.py" \
            --run_name esmfold2fast_tts \
            --gpus 0 1 2 3 \
            --seeds "${NEXT_SEEDS[@]}" \
            --complexes "${COMPLEXES[@]}"
        echo "$(date): Seeds 12-23 complete."
        break
    fi

    echo "Not done yet. Sleeping $((CHECK_INTERVAL/60)) min..."
    echo ""
    sleep "$CHECK_INTERVAL"
done
