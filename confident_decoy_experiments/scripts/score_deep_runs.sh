#!/bin/bash
# Score the two 50-seed deep runs as each finishes, and print the before/after comparison.
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "[$(date +%H:%M)] $*"; }
wait_gone(){ while :; do a=0; for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)||continue
  case "$c" in *sweep_sched.py*"$1"*) a=1;break;; esac; done; [ "$a" -eq 0 ] && break; sleep 120; done; }

for RUN in av2deep:8av2_AC chedeep:8che_BE; do
  TAG="${RUN%%:*}"; IFACE="${RUN##*:}"
  say "waiting for $TAG ($IFACE) to finish generating"
  # the chain script may not have started chedeep yet; wait for its jobs file to be in flight
  if [ "$TAG" = "chedeep" ]; then
    while [ ! -d "$EXP/artifacts/chedeep_logs" ]; do sleep 120; done
    sleep 120
  fi
  wait_gone "${TAG}_jobs"
  say "$TAG generation done -- validating templates"
  $PY $EXP/scripts/run_health.py --preflight --jobs $EXP/artifacts/${TAG}_jobs.json --seeds 50 | tail -2
  n1=$(grep -h "Found 1 templates" $EXP/artifacts/${TAG}_logs/*.log 2>/dev/null | wc -l)
  n0=$(grep -h "Found 0 templates" $EXP/artifacts/${TAG}_logs/*.log 2>/dev/null | wc -l)
  say "$TAG template loading: $n1 loaded / $n0 null"
  say "$TAG scoring (50 seeds x 20 templates)"
  $PY $EXP/scripts/score_leader20.py --jobs $EXP/artifacts/${TAG}_jobs.json \
      --out-suffix "_$TAG" --workers 10 2>&1 | tail -8
  $PY - "$IFACE" "$TAG" <<'PYEOF'
import json,sys
EXP='/home/ebrahim/confident_decoy_experiments'
iface,tag=sys.argv[1],sys.argv[2]
new=json.load(open(f'{EXP}/artifacts/leader20_scores_{tag}.json')).get(iface,{})
old=json.load(open(f'{EXP}/artifacts/leader20_scores.json')).get(iface,{})
dq=json.load(open(f'{EXP}/artifacts/dockqdonor_scores.json')).get(iface,{})
print(f"\n=== {iface}: 5 seeds -> 50 seeds ===")
print(f"  leader20  5 seeds : picked {old.get('method_dockq')} (ipTM {old.get('method_iptm')}), "
      f"pool best {old.get('best_available')}, {old.get('n_acceptable')}/{old.get('n_poses')} acceptable")
print(f"  leader20 50 seeds : picked {new.get('method_dockq')} (ipTM {new.get('method_iptm')}), "
      f"pool best {new.get('best_available')}, {new.get('n_acceptable')}/{new.get('n_poses')} acceptable")
print(f"  dockqdonor        : picked {dq.get('sel_dockq')}, pool best {dq.get('best_dockq')}, "
      f"{dq.get('n_acceptable')}/{dq.get('n')} acceptable")
print(f"  RESCUED: 5-seed {old.get('method_rescued')}  50-seed {new.get('method_rescued')}  "
      f"dockqdonor {dq.get('rescued')}")
PYEOF
  say "$TAG COMPLETE"
done
say "BOTH DEEP RUNS COMPLETE"
