#!/bin/bash
# Exit (=> notify) when 8bf4 is scored in the full-MSA N=50 run, or when the
# pipeline is clearly dead/stalled. Silence must never mean "crashed".
EXP=/home/ebrahim/confident_decoy_experiments
OUT=$EXP/artifacts/fullmsa_out
P=8bf4
stall=0; last=-1
for i in $(seq 1 720); do   # 720 * 60s = 12h ceiling
  n=$(ls -d $OUT/$P/seed_* 2>/dev/null | while read d; do \
        [ -f "$d/predictions/${P}_summary_confidence_sample_4.json" ] && echo x; done | wc -l)
  if python3 -c "import json,sys; s=json.load(open('$EXP/artifacts/fullmsa_n50_partial.json')); sys.exit(0 if '$P' in s['scored_pdbs'] else 1)" 2>/dev/null; then
    echo "DONE: 8bf4 scored — full-MSA N=50 complete for all 15 interfaces"
    python3 -c "
import json;s=json.load(open('$EXP/artifacts/fullmsa_n50_partial.json'))
for r in s['dockq']:
    if r['interface'].startswith('8bf4'): print(r)"
    exit 0
  fi
  if [ "$n" -eq "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=$n
  # 8bf4 seeds still coming? if no generation progress for 45 min AND not yet at 50, something died
  if [ "$stall" -ge 45 ] && [ "$n" -lt 50 ]; then
    echo "STALLED: 8bf4 stuck at $n/50 seeds for 45 min — generation likely died"
    pgrep -fa "fullmsa_queue_sched|inference.py" | head -5
    exit 1
  fi
  sleep 60
done
echo "TIMEOUT: 12h elapsed, 8bf4 still not scored (seeds=$last/50)"
exit 2
