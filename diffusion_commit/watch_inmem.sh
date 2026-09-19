#!/usr/bin/env bash
# Health check for the in-memory self-templating run. Every check here corresponds to a way this
# experiment has ALREADY failed silently, which is the only reason to check it:
#
#   hook fired per fold      a fold with no splice scores normally and looks like a result
#   5 templates consumed     ids unregistered / wrong date schema / dedup all yield fewer, silently
#   4 decode sets per fold    a missing observe cycle loses an entire recycle arm
#   5 conf json per decode    no json => no ipTM => only oracle DockQ comparable
#   workers alive             a dead worker just stops adding rows; nothing else complains
#   progress advancing        alive but wedged looks identical to working, from the outside
#   claims vs folds           a claimed-but-unscored unit is skipped FOREVER on resume
#
# Stop with: touch <D>/WATCH_STOP
set -u
D=${D:-/home/ebrahim/tts_abag/experimental_data/inmem}
N=${N:-13}
EXPECT=${EXPECT:-75}
GPUS=${GPUS:-"1 5 7"}
PYBIN=${PYBIN:-/home/ebrahim/Protenix/.venv/bin/python}
INTERVAL=${INTERVAL:-300}
STALL_MIN=${STALL_MIN:-20}
LOG=$D/watch.log
STOP=$D/WATCH_STOP
rm -f "$STOP"
last=-1; last_change=$(date +%s)

while :; do
  [ -f "$STOP" ] && { echo "[$(date +%H:%M:%S)] stop file seen" | tee -a "$LOG"; exit 0; }
  now=$(date +%s)
  folds=$(grep -hcE "interfaces in" "$D"/n${N}_g*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
  fires=$(grep -hc "SELFTMPL" "$D"/n${N}_g*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
  fails=$(grep -hcE "FAILED|SCORING produced no rows" "$D"/n${N}_g*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
  # Count the PYTHON processes only: the tmux wrapper shell carries the same command line, so a
  # naive grep reports double and a dead worker could hide behind its own wrapper.
  workers=$(ps -u "$USER" -o cmd= | grep -cE "^[^ ]*python[0-9.]* .*[f]old\.py --queue")
  claims=$(ls "$D/.claims${N}" 2>/dev/null | wc -l)
  [ "$folds" -ne "$last" ] && { last=$folds; last_change=$now; }
  stalled=$(( (now - last_change) / 60 ))

  msg="[$(date '+%m-%d %H:%M:%S')] folds ${folds}/${EXPECT} | hook fires ${fires} | claims ${claims} | workers ${workers}"
  for G in $GPUS; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$G" 2>/dev/null)
    msg="$msg | gpu$G ${u}%"
  done

  # ONE splice per fold. The model logs SELFTMPL only when it actually splices (`if _st_feats and
  # _st_splice`), so observation decodes are silent -- they are verified by their output dirs below,
  # not by log lines. Expecting 3 lines per fold was a false alarm on the first check.
  # IN-FLIGHT FOLDS ARE NOT FAILURES. `folds` counts COMPLETED folds (scored), but a fold that is
  # mid-run has already spliced, so fires legitimately exceeds folds by up to one per worker. Flag
  # only when fires is below folds (a fold that scored without splicing) or above the in-flight
  # ceiling. Comparing the two directly produced a false alarm on every check.
  ceil=$(( folds + workers ))
  { [ "$fires" -lt "$folds" ] || [ "$fires" -gt "$ceil" ]; } && \
    msg="$msg | !! $fires splices for $folds done + $workers in flight"
  [ "${fails:-0}" -gt 0 ] && msg="$msg | !! $fails FAILURES"
  [ "$stalled" -ge "$STALL_MIN" ] && [ "$folds" -lt "$EXPECT" ] && msg="$msg | !! NO PROGRESS ${stalled}min"
  [ "$workers" -eq 0 ] && [ "$folds" -lt "$EXPECT" ] && msg="$msg | !! NO WORKERS with work left"

  # a fold must leave 3 mid-loop decode dirs, each with 5 poses and 5 confidence json
  bad_dec=0; bad_conf=0
  for r in "$D"/n${N}/free/*/*/*/; do
    # Only FINISHED folds: run.json is written after the fold returns, so its absence means the fold
    # is still running and its later decode dirs do not exist yet.
    [ -f "$r/run.json" ] || continue
    [ -d "$r/_hook" ] || continue
    nd=$(ls -d "$r"/_hook/c* 2>/dev/null | wc -l)
    [ "$nd" -eq 3 ] || bad_dec=$((bad_dec+1))
    for c in "$r"/_hook/c*; do
      [ "$(ls "$c"/poses/*summary_confidence*.json 2>/dev/null | wc -l)" -eq 5 ] || bad_conf=$((bad_conf+1))
    done
  done
  [ "$bad_dec" -gt 0 ] && msg="$msg | !! $bad_dec folds without 3 decode sets"
  [ "$bad_conf" -gt 0 ] && msg="$msg | !! $bad_conf decodes missing confidence json"

  # template count: the driver asserts it, but surface it so a systematic drop is visible early
  low=$(grep -hoE "Found [0-9]+ templates" "$D"/n${N}_g*.log 2>/dev/null | awk '{print $2}' \
        | sort -n | uniq -c | awk '$2<5 && $2>0 {print $2":"$1}' | paste -sd' ')
  [ -n "$low" ] && msg="$msg | template counts below 5 seen: $low (4 = the stock PDB templates, expected)"

  # REMEDIATE LEAKED CLAIMS. fold.py reconciles only at startup, so a worker killed mid-fold leaves
  # its unit claimed-but-unscored and it is skipped for the rest of the pass. That happened for real:
  # killing the GPU-7 worker to free a card stranded 9uzh seed 300011. Release any claim that is
  # neither scored nor held by a live worker; the unit is then picked up by the next pass.
  rel=$(ls "$D/.claims${N}" 2>/dev/null | wc -l)
  if [ "$rel" -gt $(( folds + workers )) ]; then
    "$PYBIN" - "$D" "$N" "$workers" <<'EOF' >> "$LOG" 2>&1
import csv, os, sys, collections
d, n, live = sys.argv[1], sys.argv[2], int(sys.argv[3])
scored = collections.defaultdict(set)
for r in csv.DictReader(open(f"{d}/n{n}.csv")):
    scored[r["iface"].split("_")[0]].add(r["seed"])
cl = f"{d}/.claims{n}"
stale = [c for c in os.listdir(cl)
         if c.split("_")[-1] not in scored.get(c.split("_")[0], set())]
# leave `live` of them alone: those are the folds currently in flight, newest first
stale.sort(key=lambda c: os.path.getmtime(f"{cl}/{c}"))
for c in stale[:max(0, len(stale) - live)]:
    os.rmdir(f"{cl}/{c}")
    print(f"  [watch] released leaked claim {c}")
EOF
  fi
  # SCORE AS WE GO. Run opportunistically rather than by hand: scoring drifted 442 pose sets behind
  # the folding, which made two interfaces look missing from the results when they had in fact
  # finished, and made a third report on 10 of 25 seeds. Skipped while a scorer is already running so
  # the passes cannot pile up on the CPU.
  if ! ps -u "$USER" -o cmd= | grep -q "[s]core_hook_poses.py"; then
    nohup "$PYBIN" /home/ebrahim/tts_abag/diffusion_commit/score_hook_poses.py \
      --root "$D/n${N}" --csv "$D/decodes.csv" --jobs 6 >> "$D/score.log" 2>&1 &
  fi
  echo "$msg" | tee -a "$LOG"

  if [ "$folds" -ge "$EXPECT" ]; then
    echo "[$(date +%H:%M:%S)] ALL $EXPECT FOLDS DONE (fires=$fires fails=$fails)" | tee -a "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
