#!/usr/bin/env bash
# The PAIRED FREE ARM: 25 more plain folds per interface (seeds 300050-300074), taking the free pool
# from 50 to 75 seeds. No templates, nothing carried over -- ordinary inference.
#
# WHY IT MATTERS. Without this the gen-2 comparison is not budget-matched: 25 free + 25 gen-1 + 25 gen-2
# is 75 seeds of compute, so comparing it against a 50-seed free pool would credit self-templating for
# what is really just more sampling. Every interface currently has 50 free seeds (a few have 59 or 70),
# and NONE has 75.
#
# Usage: GPU=6 IFACES_FILE=ifaces291.txt bash gen2_free.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
IFACES_FILE=${IFACES_FILE:-$DC/ifaces291.txt}
P=$REPO/artifacts/pxm2025
CSV=$REPO/artifacts/commit11/results.csv
OUT=${OUT:-$REPO/artifacts/free75}
LOG=${LOG:-$REPO/artifacts/gen2_free.log}
STOP=${STOP:-$REPO/artifacts/FREE75_STOP}
NS=${NS:-5}; NCYCLE=${NCYCLE:-10}
SEEDS=${SEEDS:-"$(seq 300050 300074 | tr '\n' ' ')"}
rm -f "$STOP"; mkdir -p "$OUT"; cd "$REPO" || exit 1
echo "[$(date '+%m-%d %H:%M:%S')] FREE75 START gpu$GPU $(wc -l < "$IFACES_FILE") interfaces" | tee -a "$LOG"

while read -r IFACE; do
  [ -n "$IFACE" ] || continue
  [ -f "$STOP" ] && { echo "[$(date '+%H:%M:%S')] stop file seen" | tee -a "$LOG"; exit 0; }
  PDB=${IFACE%%_*}
  JOB=$P/jsons/${PDB}-final-updated.json
  REF=$P/cif/$PDB.cif
  [ -s "$JOB" ] && [ -s "$REF" ] || { echo "  !! $IFACE missing inputs"; continue; }
  # Only interfaces short of 75 free seeds. Counted from the pool csv PLUS what this arm has already
  # written, so a resume does not redo work.
  HAVE=$($PY - "$IFACE" <<'EOF'
import csv, sys, collections
iface=sys.argv[1]; pdb,ch=iface.split('_'); c1,c2=ch[0],ch[1]
s=set()
for r in csv.DictReader(open('artifacts/pxm2025/dockq_pxm2025.csv')):
    if r['pdb']==pdb and r['chain1']==c1 and r['chain2']==c2: s.add(r['seed'])
try:
    for r in csv.DictReader(open('artifacts/commit11/results.csv')):
        if r['iface']==iface and r['arm']=='free75': s.add(r['seed'])
except FileNotFoundError: pass
print(len(s))
EOF
)
  if [ "${HAVE:-0}" -ge 75 ]; then
    echo "[$(date '+%H:%M:%S')]     $IFACE already has $HAVE free seeds, skipping" | tee -a "$LOG"; continue
  fi
  echo "[$(date '+%m-%d %H:%M:%S')] >>> $IFACE (has $HAVE free seeds)" | tee -a "$LOG"

  for SEED in $SEEDS; do
    grep -q "^${IFACE},${SEED},free75," "$CSV" 2>/dev/null && continue
    W=$OUT/$IFACE/s$SEED
    mkdir -p "$W/out"
    cd /home/ebrahim/Protenix || exit 1
    env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
    $PY runner/inference.py --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$JOB" --dump_dir "$W/out" --seeds "$SEED" \
      --model.N_cycle $NCYCLE --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
      --use_template true > "$W/run.log" 2>&1
    cd "$REPO" || exit 1
    N=$(find "$W/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
    if [ "$N" -ne "$NS" ]; then
      echo "  !! $IFACE s$SEED: $N/$NS poses -- not scoring"
      grep -m1 -E "ERROR|Error" "$W/run.log" 2>/dev/null | cut -c1-130 | sed 's/^/     /'
      rm -rf "$W/out"; continue
    fi
    $PY "$DC/row.py" --dir "$W" --iface "$IFACE" --seed "$SEED" --arm free75 \
        --donor -1 --ref "$REF" --scorer pxm >> "$CSV"
    rm -rf "$W/out"
  done
  echo "[$(date '+%m-%d %H:%M:%S')]     $IFACE: $(grep -c "^$IFACE,.*,free75," "$CSV" 2>/dev/null) free75 rows" | tee -a "$LOG"
done < "$IFACES_FILE"
echo "[$(date '+%m-%d %H:%M:%S')] FREE75 COMPLETE" | tee -a "$LOG"
