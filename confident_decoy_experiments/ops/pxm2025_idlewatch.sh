#!/usr/bin/env bash
# Exit as soon as a PXM-2025 worker is genuinely idle, so the seed-chunked runner can be swapped in.
#
# "Idle" is NOT a momentary 0% reading -- every complex spends 1-3 minutes in CPU template search
# before touching the GPU, and 9lad showed 0% for ~3 min at startup. So a low-utilisation reading
# only counts if it persists across 3 consecutive checks 5 minutes apart. The unambiguous signals
# (a worker printing ALL DONE, or its tmux session disappearing) fire immediately.
set -u
SESS="pxm0 pxm1 pxm2 pxm6"
low=0
while true; do
  for s in $SESS; do
    if ! tmux has-session -t "$s" 2>/dev/null; then
      echo "IDLE: session $s is gone  $(date +%H:%M)"; exit 0
    fi
    if tmux capture-pane -t "$s" -p 2>/dev/null | grep -q "ALL DONE"; then
      echo "IDLE: $s exhausted the queue (ALL DONE)  $(date +%H:%M)"; exit 0
    fi
  done
  # Match the python processes ONLY. A bare `pgrep -fc runner/inference.py` also counts any shell
  # whose command line mentions the path (an interactive status check did exactly that, reporting
  # 5/4), which would mask a genuinely dead worker.
  n=$(ps -u "$USER" -o args= \
      | grep -c "^/home/ebrahim/Protenix/.venv/bin/python runner/inference.py" || true)
  if [ "$n" -lt 4 ]; then low=$((low+1)); else low=0; fi
  if [ "$low" -ge 3 ]; then
    echo "IDLE: only $n/4 inference processes for 15 min  $(date +%H:%M)"; exit 0
  fi
  sleep 300
done
