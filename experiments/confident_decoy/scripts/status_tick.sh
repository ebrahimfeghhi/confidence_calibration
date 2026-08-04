#!/bin/bash
# Sleep ~5 min, then print a compact status snapshot of the full-MSA N=4 round and exit.
# Run in background; the harness notifies on exit, which is the 5-minute status ping.
EXP=/home/ebrahim/confident_decoy_experiments
OUT=$EXP/artifacts/fullmsa_out
LOGD=$EXP/artifacts/fullmsa_logs
sleep 300
echo "=== STATUS $(date +%H:%M:%S) ==="
if [ -f "$EXP/artifacts/fullmsa_n50_report.json" ]; then
  echo "REPORT READY"
else
  echo -n "seeds 101-150 per pdb: "
  for p in 7z2m 8slb 8yxi 8the 8fas 8ux6 8fbw 8q3j 8che 8av2 8aok 8bf4; do
    n=0
    for s in $(seq 101 150); do
      [ -f "$OUT/$p/seed_$s/predictions/${p}_summary_confidence_sample_4.json" ] && n=$((n+1))
    done
    printf "%s:%s/50 " "$p" "$n"
  done; echo
  echo "scheduler: $(tail -1 "$LOGD/queue50.log" 2>/dev/null | cut -c1-90)"
  echo "scorer:    $(tail -1 "$LOGD/n50_report.log" 2>/dev/null | cut -c1-90)"
  echo -n "GPU mem: "; nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -4 | tr '\n' ' '; echo
  echo -n "cores: "; ps -eo pcpu,cmd | grep -E "inference.py|/ost " | grep -v grep | awk '{c+=$1} END{printf "~%.0f\n", c/100}'
fi
