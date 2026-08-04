#!/bin/bash
# Score interfaces AS THEY COMPLETE, instead of waiting for a whole 25-job arm to finish.
#
# A 25-interface arm takes 4-8 h, but individual interfaces finish continuously throughout. Scoring is
# CPU/OST only, so it costs nothing on the GPUs and turns a long blind wait into a running result.
# It also surfaces a broken arm early: if the first few interfaces come back with templates that never
# loaded, or with n<250 poses, that is visible in minutes rather than after the whole arm has run.
#
# Each pass: for every arm, find interfaces with all 50 seeds that are NOT yet in <arm>_scores.json,
# write them to a filtered jobs file, and score just those. score_arm.py loads the existing scores file
# and adds to it, and its OST cache makes any re-scoring near-free.
#
# The template guard inside score_arm.py still applies per pass, so a null arm is refused here too.
#
# Usage: setsid nohup bash scripts/score_watcher.sh > artifacts/decoy_logs/score_watcher.log 2>&1 &
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "[$(date +%H:%M)] $*"; }

# arm : jobs file : output root
ARMS=(
  "topiptm26:$EXP/artifacts/topiptm26_jobs25.json:$EXP/artifacts/sweep_out/topiptm26"
  "ifacelddt26:$EXP/artifacts/ifacelddt26_jobs25.json:$EXP/artifacts/sweep_out/ifacelddt26"
  "globallddt26:$EXP/artifacts/globallddt26_jobs25.json:$EXP/artifacts/sweep_out/globallddt26"
)

while :; do
  pending_total=0
  for entry in "${ARMS[@]}"; do
    ARM="${entry%%:*}"; rest="${entry#*:}"; JOBS="${rest%%:*}"; OUT="${rest##*:}"
    [ -f "$JOBS" ] || continue

    NEW=$($PY - "$ARM" "$JOBS" "$OUT" <<'PY'
import json,glob,os,sys
arm,jobs_f,out=sys.argv[1],sys.argv[2],sys.argv[3]
EXP="/home/ebrahim/confident_decoy_experiments"
jobs=json.load(open(jobs_f))
sf=f"{EXP}/artifacts/{arm}_scores.json"
scored=set(json.load(open(sf))) if os.path.exists(sf) else set()
new=[]
for j in jobs:
    if j["interface"] in scored: continue
    d=f"{out}/{j['run']}/{j['name']}"
    if len(glob.glob(f"{d}/seed_*/predictions/{j['name']}_summary_confidence_sample_4.json"))>=50:
        new.append(j)
p=f"{EXP}/artifacts/{arm}_jobs_newly_complete.json"
json.dump(new,open(p,"w"),indent=1)
print(f"{len(new)} {len(scored)} {len(jobs)}")
PY
)
    n_new=$(echo "$NEW" | cut -d' ' -f1); n_scored=$(echo "$NEW" | cut -d' ' -f2); n_tot=$(echo "$NEW" | cut -d' ' -f3)
    pending_total=$((pending_total + n_tot - n_scored - n_new))
    if [ "${n_new:-0}" -gt 0 ]; then
      say "$ARM: $n_new newly complete (had $n_scored/$n_tot) -- scoring"
      $PY $EXP/scripts/score_arm.py --arm "$ARM" \
          --jobs $EXP/artifacts/${ARM}_jobs_newly_complete.json \
          --out "$OUT" --logs $EXP/artifacts/${ARM}_logs --workers 10 2>&1 | grep -v "^$" | tail -20
      n=$($PY -c "import json;d=json.load(open('$EXP/artifacts/${ARM}_scores.json'));print(f'{sum(1 for v in d.values() if v[\"rescued\"])}/{len(d)}')" 2>/dev/null)
      say "$ARM running tally: $n rescued"
    fi
  done

  # stop once every arm is fully scored and no scheduler is left running
  sched=0
  for p in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$c" in *sweep_sched.py*) sched=1; break;; esac
  done
  if [ "$sched" -eq 0 ] && [ "$pending_total" -le 0 ]; then
    say "all arms scored and no scheduler running -- watcher exiting"; break
  fi
  sleep 600
done
