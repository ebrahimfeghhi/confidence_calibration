#!/usr/bin/env bash
# Feed ALL 250 free-run poses of one interface into the template embedder at once.
#
# WHAT THE EMBEDDER ACTUALLY DOES WITH THEM: pairformer.py loops `for template_id in range(num_templates)`
# and then `u = u / num_templates`. It AVERAGES. So 250 templates is not 250 hypotheses offered for
# selection -- it is one mean contact pattern, weighted by how many poses sit on each surface. On
# 9i20_AB that mean is 229 decoy poses against 21 acceptable ones, so the prediction is that this
# ENTRENCHES the decoy. Running it because that is falsifiable, not because it is expected to work.
#
# Memory is flat in template count (the loop is sequential), but COMPUTE is not: the embedder runs every
# recycle, so 250 templates x 10 recycles is ~10-20x a normal fold's template work.
#
# MAX_TEMPLATES must be raised from its trained default of 4, and MAX_TEMPLATE_CANDIDATES from 20 --
# that second cap truncates the pool AFTER dedup and BEFORE selection, so raising only the first
# silently keeps 20 templates and the run looks fine.
#
# Usage: GPU=1 IFACE=9i20_AB NPOSE=250 bash allpose250.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}; IFACE=${IFACE:?}
PDB=${IFACE%%_*}
NPOSE=${NPOSE:-250}
SEED=${SEED:-900001}          # a fresh seed: this is a new fold, not a continuation
NCYCLE=${NCYCLE:-10}; NS=${NS:-5}
P=$REPO/artifacts/pxm2025
CSV=$REPO/artifacts/commit11/results.csv
D=$REPO/artifacts/commit11/allpose250/${IFACE}_n${NPOSE}
mkdir -p "$D/out"; cd "$REPO" || exit 1

POSES=$(find $P/out/$PDB/$PDB -name "${PDB}_sample_*.cif" | sort | head -$NPOSE)
N=$(echo "$POSES" | grep -c .)
[ "$N" -eq "$NPOSE" ] || { echo "!! found $N poses, wanted $NPOSE"; exit 1; }
TIDS=""; i=0
for f in $POSES; do TIDS="$TIDS $(printf 'z%03d' $i)"; i=$((i+1)); done
echo "=== $(date +%H:%M:%S) GPU$GPU $IFACE: $N poses as templates, $NCYCLE recycles, seed $SEED"

if [ ! -s "$D/tmpl/job.json" ]; then
  SELFTMPL_JOBS=$P/jsons $PY "$DC/build_selftmpl_general.py" --pdb "$PDB" --out "$D/tmpl" \
    --poses $POSES --tids $TIDS > "$D/build.log" 2>&1
fi
[ -s "$D/tmpl/job.json" ] || { echo "!! template build FAILED"; tail -5 "$D/build.log" | sed 's/^/   /'; exit 1; }
echo "    templates in job: $($PY -c "
import json;d=json.load(open('$D/tmpl/job.json'));o=d[0] if isinstance(d,list) else d
import itertools
n=0
for s in o.get('sequences',[]):
    for v in s.values():
        if isinstance(v,dict) and v.get('templates'): n=max(n,len(v['templates']))
print(n)")"

cd /home/ebrahim/Protenix || exit 1
env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
    PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 \
    PROTENIX_MAX_TEMPLATES=$NPOSE PROTENIX_MAX_TEMPLATE_CANDIDATES=$((NPOSE+50)) \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$D/tmpl/job.json" --dump_dir "$D/out" --seeds "$SEED" \
  --model.N_cycle $NCYCLE --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
  --use_template true > "$D/run.log" 2>&1
cd "$REPO" || exit 1

# ASSERT the templates actually arrived. "Found N templates" well below NPOSE means a cap bit and the
# run is not the experiment -- it would still produce clean-looking numbers.
grep -oE 'Found [0-9]+ templates' "$D/run.log" | sort | uniq -c | sed 's/^/    /'
grep -oE 'cycle=[0-9]+' "$D/run.log" | head -1 | sed 's/^/    /'
NC=$(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
if [ "$NC" -ne "$NS" ]; then
  echo "    !! produced $NC/$NS poses -- not scoring, see $D/run.log"
  grep -m3 -E "ERROR|Error|out of memory" "$D/run.log" | cut -c1-150 | sed 's/^/       /'; exit 1
fi
$PY "$DC/row.py" --dir "$D" --iface "$IFACE" --seed "$SEED" --arm "all${NPOSE}" \
    --donor -1 --ref "$P/cif/$PDB.cif" --scorer pxm >> "$CSV"
echo "    all${NPOSE}: $NC poses scored"
grep "^${IFACE},${SEED},all${NPOSE}," "$CSV" | awk -F, '{printf "      sample %s  dockq %s  iptm %s\n",$5,$6,$7}'
echo "=== $(date +%H:%M:%S) done"
