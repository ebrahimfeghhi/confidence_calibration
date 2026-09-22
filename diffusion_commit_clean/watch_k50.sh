#!/usr/bin/env bash
# Watch the top-5 self-templating sweep: score the arms, check health, and FIX what it can.
#
# A watcher that only prints is useless overnight -- by morning it has logged the same failure 16
# times and nothing has run. This one restarts a dead worker on its own GPU, which also releases that
# worker's stale claims, since fold.py reconciles claims at startup. Anything it cannot fix it shouts
# about in the log with a line starting "ESCALATE", so a single grep finds it.
#
#   watch_top5.sh [interval_seconds]
set -u
T=/home/ebrahim/tts_abag/experimental_data/k50
CODE=/home/ebrahim/tts_abag/diffusion_commit
PY=/home/ebrahim/Protenix/.venv/bin/python
GPUS=(1 5 6)
INT=${1:-1800}

while true; do
  echo "================ $(date '+%F %H:%M') ================"

  # ---- health: one worker per GPU, restarted if gone -------------------------------------------
  total=$(wc -l < "$T/queue.tsv")
  claimed=$(ls "$T/claims" 2>/dev/null | wc -l)
  if [ "$claimed" -lt "$total" ]; then
    for i in 0 1 2; do
      if ! pgrep -u "$USER" -f "top5/queue.tsv" | xargs -r ps -o pid=,cmd= -p 2>/dev/null \
           | grep -q "CUDA_VISIBLE_DEVICES=${GPUS[$i]}" 2>/dev/null; then
        if ! tmux has-session -t "k50w$i" 2>/dev/null; then
          echo "REMEDIATE: worker $i (GPU ${GPUS[$i]}) gone -- restarting"
          tmux new-session -d -s "k50w$i" \
            "cd $CODE && CUDA_VISIBLE_DEVICES=${GPUS[$i]} LAYERNORM_TYPE=openfold $PY fold.py \
             --queue $T/queue.tsv --out $T/out --csv $T/results.csv --templates $T/ns \
             --claim-dir $T/claims --drop-cifs 2>&1 | tee -a $T/w$i.log"
        fi
      fi
    done
  fi
  alive=$(pgrep -u "$USER" -f "top5/queue.tsv" | wc -l)
  echo "workers=$alive  claimed=$claimed/$total  rows=$(( $(wc -l < "$T/results.csv" 2>/dev/null || echo 1) - 1 ))"

  # ---- errors since the last check --------------------------------------------------------------
  for i in 0 1 2; do
    # grep -c PRINTS 0 and EXITS 1 when nothing matches, so `|| echo 0` yields "0\n0" and every
    # numeric test below dies. Swallow the exit code instead of appending a second value.
    n=$(grep -cE "Traceback|OutOfMemory|CUDA out of memory" "$T/w$i.log" 2>/dev/null; true)
    n=${n:-0}
    [ "$n" -gt 0 ] && echo "ESCALATE: worker $i has $n error lines (grep -nE 'Traceback|out of memory' $T/w$i.log)"
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,5,6 \
    | sed 's/^/  gpu /'

  # ---- the actual science: per-arm DockQ, split by interface class ------------------------------
  $PY - <<'PYEOF'
import csv, collections, os, sys
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean")
import numpy as np
from separation import pool_classes
from fmt import table
R = "/home/ebrahim/tts_abag/experimental_data/k50/results.csv"
if not os.path.exists(R):
    raise SystemExit("  no results yet")
cls = pool_classes()
# best DockQ per (interface, arm, seed) -- a fold's 5 samples are one unit of work
best = collections.defaultdict(float)
for r in csv.DictReader(open(R)):
    k = (r["iface"], r["arm"], r["seed"])
    best[k] = max(best[k], float(r["dockq"]))
by = collections.defaultdict(list)
for (iface, arm, seed), d in best.items():
    by[(cls.get(iface, "?"), arm)].append(d)
rows = []
for k in sorted(by, key=lambda k: ({"decoy": 0, "solved": 1, "nocov": 2}.get(k[0], 3), k[1])):
    v = by[k]
    rows.append([k[0], k[1], len(v), round(float(np.mean(v)), 3),
                 round(float(np.median(v)), 3), sum(1 for x in v if x >= 0.23)])
print(table(["class", "arm", "folds scored (n)", "mean best DockQ", "median", "acceptable"],
            rows, align="l l r r r r"))
# the headline contrast, on the interfaces where BOTH arms have finished
pair = collections.defaultdict(dict)
for (iface, arm, seed), d in best.items():
    pair[(iface, seed)][arm] = d
both = [(cls.get(i, "?"), v["oracle"], v["conf"])
        for (i, s), v in pair.items() if "oracle" in v and "conf" in v]
if both:
    out = []
    for c in ("decoy", "solved", "nocov"):
        s = [(o, cf) for k, o, cf in both if k == c]
        if s:
            out.append([c, len(s), round(float(np.mean([o for o, _ in s])), 3),
                        round(float(np.mean([cf for _, cf in s])), 3),
                        round(float(np.mean([o - cf for o, cf in s])), 3)])
    print(table(["class", "paired folds (n)", "oracle", "conf", "oracle - conf"],
                out, align="l r r r r"))
PYEOF

  # ---- the selection table: what ipTM actually ships, before vs after ---------------------------
  # Regenerated every cycle so the morning table is current, and written to a file as well as the log
  # so it can be read without scrolling through 16 cycles of health output.
  cd /home/ebrahim/tts_abag/diffusion_commit_clean 2>/dev/null && \
    cd /home/ebrahim/tts_abag/diffusion_commit_clean && $PY k50_report.py > "$T/table.txt" 2>&1
  tail -25 "$T/table.txt"

  sleep "$INT"
done
