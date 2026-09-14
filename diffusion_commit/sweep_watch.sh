#!/usr/bin/env bash
# Report the free-run sweep every 5 minutes: progress, the straddle rate, and anything wrong.
#
# WHAT IT REPORTS AND WHY. The sweep's purpose is to find seeds whose 5 donor poses STRADDLE DockQ
# 0.23 -- one seed yielding both a good pose and a decoy. Only those can test whether committing the
# trunk to a pose lets confidence rank them; a seed whose poses are all good (or all bad) has a flat
# donor axis and says nothing either way. So the headline number here is the straddle count, not the
# mean DockQ.
#
# IT CHECKS FOR TROUBLE, NOT JUST PROGRESS: a dead worker, a held GPU that lost its holder, folds that
# produced no poses, and Tracebacks in the logs. A watcher that only prints progress lets a wedged run
# burn a card unnoticed -- that has happened here for 16 hours.
#
# Usage: bash sweep_watch.sh            (stop: touch the stop file it prints)
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
CSV=$REPO/artifacts/commit11/results.csv
IFACE=${IFACE:-8fas_BC}
WANT=${WANT:-100}
PERIOD=${PERIOD:-300}
GPUS=${GPUS:-"5 6"}
LOG=${LOG:-$REPO/artifacts/sweep_watch.log}
STOP=${STOP:-$REPO/artifacts/SWEEP_WATCH_STOP}
PY=/home/ebrahim/Protenix/.venv/bin/python
rm -f "$STOP"
echo "[sweep_watch] $IFACE every ${PERIOD}s; log $LOG; stop: touch $STOP"

while true; do
  [ -f "$STOP" ] && { echo "[sweep_watch] stop file seen, exiting"; exit 0; }
  {
    echo "===== $(date '+%m-%d %H:%M:%S')  $IFACE ====="

    # --- progress + straddle, deduped on read (exact dupes exist from before score() had its guard)
    $PY - "$CSV" "$IFACE" "$WANT" <<'EOF'
import csv, sys, collections, statistics as st
csvf, iface, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen, by = set(), collections.defaultdict(dict)
try:
    for r in csv.DictReader(open(csvf)):
        if r["iface"] != iface or r["arm"] != "donor":
            continue
        key = (r["seed"], r["sample"], r["dockq"])
        if key in seen:            # exact duplicate row -> ignore
            continue
        seen.add(key)
        by[r["seed"]][r["sample"]] = float(r["dockq"])
except FileNotFoundError:
    print("  no results.csv yet"); raise SystemExit
full = {s: list(v.values()) for s, v in by.items() if len(v) >= 2}
if not full:
    print("  no scored seeds yet"); raise SystemExit
strad = [s for s, v in full.items() if min(v) < 0.23 <= max(v)]
good  = [s for s, v in full.items() if max(v) >= 0.23]
dead  = [s for s, v in full.items() if max(v) < 0.23]
allp  = [x for v in full.values() for x in v]
print(f"  seeds scored {len(full)}/{want}   poses {len(allp)}")
print(f"  STRADDLE {len(strad)} ({len(strad)/len(full)*100:.0f}%)   "
      f"all-good {len(good)-len(strad)}   all-bad {len(dead)}")
print(f"  pose DockQ: mean {st.mean(allp):.3f}  max {max(allp):.3f}  "
      f"acceptable {sum(1 for x in allp if x>=0.23)/len(allp)*100:.0f}%")
if strad:
    ex = sorted(strad, key=lambda s: max(full[s]) - min(full[s]), reverse=True)[:5]
    print("  widest straddlers (arm candidates): " +
          ", ".join(f"{s}[{min(full[s]):.2f}-{max(full[s]):.2f}]" for s in ex))
EOF

    # --- workers alive?
    for g in $GPUS; do
      w=$(tmux has-session -t "b$g" 2>/dev/null && echo up || echo DOWN)
      h=$(tmux has-session -t "hhold$g" 2>/dev/null && echo held || echo UNHELD)
      u=$(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
            ps -o user= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u | tr '\n' ',' )
      flag=""
      [ "$w" = "DOWN" ] && flag="$flag  <-- WORKER GONE"
      [ "$h" = "UNHELD" ] && flag="$flag  <-- NOT HELD"
      echo "  GPU$g worker=$w holder=$h users=${u:-none}$flag"
    done

    # --- folds that produced nothing. A fold IN PROGRESS also has zero poses, so age it out: a fold
    # takes ~2 min, and flagging in-flight ones made the watcher cry wolf on its first report. Only a
    # fold whose log has gone quiet for STALE_MIN minutes, or that logged an error, is a real failure.
    STALE_MIN=${STALE_MIN:-6}
    empty=""
    for d in "$REPO"/artifacts/commit11/${IFACE}_s*/; do
      [ -d "$d/donor/out" ] || continue
      [ "$(find "$d/donor/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)" -eq 0 ] || continue
      if grep -qE "KeyError|Traceback|ERROR" "$d/fold.log" 2>/dev/null; then
        empty="$empty $(basename "$d")(error)"
      elif [ -z "$(find "$d/fold.log" -mmin -$STALE_MIN 2>/dev/null)" ]; then
        empty="$empty $(basename "$d")(stalled)"
      fi
    done
    [ -n "$empty" ] && echo "  !! folds with ZERO poses:$empty"
    tb=$(grep -l "Traceback" /tmp/free8fas_g*.log 2>/dev/null | tr '\n' ' ')
    [ -n "$tb" ] && echo "  !! Traceback in: $tb"
    echo
  } >> "$LOG" 2>&1
  sleep "$PERIOD"
done
