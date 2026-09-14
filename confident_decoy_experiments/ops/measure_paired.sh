#!/usr/bin/env bash
# Log the PAIRED-row fraction of the merged MSA for all 49 complexes.
#
# WHY: msa_featurizer.py:333 builds the MSA row axis as concatenate([msa_all_seq, msa]) -- paired
# (cross-chain, species-matched) rows on top, per-chain unpaired rows below. The three-phase run's
# middle phase drops that top block, so the block's SIZE is exactly how much that phase differs from
# the full-MSA phase. On 8qf4 it was 125 of 11575 rows = 1.1%, which would make the middle phase
# nearly a no-op. This measures whether that is true across the set before we spend days of GPU.
#
# HOW: N_cycle=1 with PROTENIX_MSA_PHASES=0,1 makes cycle 0 the intra-chain phase, so the masking
# code runs and prints its counts on the very first cycle, then inference finishes. One seed, one
# sample -- the pose is irrelevant, we only want the log line. ~15-20s per complex.
#
# Absolute --out path on purpose: sweep_sched runs inference with cwd=/home/ebrahim/Protenix, so a
# relative path writes outside tts_abag (which a smoke test already did once).
set -o pipefail

REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=${GPU:-0}
OUT=$A/paired_fraction.csv
LOGD=$A/paired_logs
mkdir -p "$LOGD" "$A/paired_out"
cd "$REPO" || exit 1

echo "pdb,n_paired,n_rows,frac_paired" > "$OUT"
n=0
for pdb in $(awk '{print $1}' ops/sweep_49_order.txt); do
  n=$((n+1))
  jf=$A/paired_job_${pdb}.json
  python3 - "$pdb" "$jf" <<'PYEOF'
import json,sys
p,jf=sys.argv[1],sys.argv[2]
json.dump([{"run":f"paired_{p}","pdb":p,
            "json":f"/home/ebrahim/Protenix/foldbench_abag/jsons/{p}-final-updated.json",
            "name":p,"seeds":[901]}], open(jf,"w"))
PYEOF
  PROTENIX_MSA_PHASES=0,1 PROTENIX_MSA_PHASE_DEBUG=1 \
    $PY scripts/sweep_sched.py --jobs "$jf" --out "$A/paired_out" --logs "$LOGD" \
        --status "$A/paired_status.json" --gpus "$GPU" --n-sample 1 \
        --extra-args '--model.N_cycle 1' >/dev/null 2>&1

  L=$LOGD/paired_${pdb}_${pdb}.log
  line=$(grep -oP "dropped \K\d+ paired rows of \d+" "$L" 2>/dev/null | head -1)
  if [ -n "$line" ]; then
    np=$(echo "$line" | awk '{print $1}'); nr=$(echo "$line" | awk '{print $5}')
    echo "$pdb,$np,$nr,$(awk -v a=$np -v b=$nr 'BEGIN{printf "%.4f",a/b}')" >> "$OUT"
    printf "  [%2d/49] %-6s %6d / %-7d %5.1f%%\n" "$n" "$pdb" "$np" "$nr" \
      "$(awk -v a=$np -v b=$nr 'BEGIN{print 100*a/b}')"
  else
    noop=$(grep -oP "NO-OP: n_paired=\K\d+ n_rows=\d+" "$L" 2>/dev/null | head -1)
    if [ -n "$noop" ]; then
      np=$(echo "$noop"|awk '{print $1}'); nr=$(echo "$noop"|sed 's/.*n_rows=//')
      echo "$pdb,$np,$nr,0" >> "$OUT"; printf "  [%2d/49] %-6s NO-OP (n_paired=%s n_rows=%s)\n" "$n" "$pdb" "$np" "$nr"
    else
      echo "$pdb,NA,NA,NA" >> "$OUT"; printf "  [%2d/49] %-6s FAILED (see %s)\n" "$n" "$pdb" "$L"
    fi
  fi
  rm -f "$jf"
done
echo "### done -> $OUT"
