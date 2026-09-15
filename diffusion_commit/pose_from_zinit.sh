#!/usr/bin/env bash
# Score a pose through a pair rep seeded by THAT POSE ALONE: one recycle from z_init, the pose as the
# only template, no trunk carried over.
#
# WHY. Within a seed all five poses share one trunk state, and ipTM is a property of that state -- the
# median within-seed ipTM SD is 0.0078 while DockQ can span the whole scale, so confidence cannot rank
# them. Here each pose gets a pair rep derived from itself: z = z_init + templateEmbedder(pose), one
# recycle. If the resulting ipTM is higher for a good pose than a decoy from the SAME seed, confidence
# has become a pose property rather than a seed property.
#
# WHAT THIS IS NOT. It does not run the confidence head on the input pose's own coordinates -- it
# decodes fresh poses from that pair rep and scores those. Commitment is strong (r = 0.92-0.985), so the
# decoded poses sit in the input pose's basin, but the number is confidence in the BASIN, not in the
# pose. Scoring the pose itself needs the confidence head called outside the normal forward.
#
# NO LOAD_TRUNK on purpose: the point is to start from the prior, not from a committed trunk.
#
# Usage: GPU=6 IFACE=9nn9_AB SEED=300004 POSES="1 4" bash pose_from_zinit.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}; IFACE=${IFACE:?}; SEED=${SEED:?}; POSES=${POSES:?}
PDB=${PDB:-${IFACE%%_*}}
NCYCLE=${NCYCLE:-1}; NS=${NS:-5}
P=$REPO/artifacts/pxm2025
REF=$P/cif/$PDB.cif
CSV=$REPO/artifacts/commit11/results.csv
OUT=$REPO/artifacts/zinit_tmpl/${IFACE}_s${SEED}
mkdir -p "$OUT"; cd "$REPO" || exit 1
SRC=$P/out/$PDB/$PDB/seed_$SEED/predictions

for k in $POSES; do
  POSE=$SRC/${PDB}_sample_${k}.cif
  [ -s "$POSE" ] || { echo "!! pose $k missing"; continue; }
  ARM="zin_p$k"
  grep -q "^${IFACE},${SEED},${ARM}," "$CSV" 2>/dev/null && { echo "=== $ARM already scored"; continue; }
  D=$OUT/p$k; mkdir -p "$D/out"
  echo "=== $(date +%H:%M:%S) GPU$GPU $IFACE s$SEED pose $k -> $ARM  ($NCYCLE recycle from z_init)"
  if [ ! -s "$D/tmpl/job.json" ]; then
    SELFTMPL_JOBS=$P/jsons $PY "$DC/build_selftmpl_general.py" --pdb "$PDB" --out "$D/tmpl" \
      --poses "$POSE" --tids "$(printf 'q%01d%02d' "$k" $((SEED % 100)))" > "$D/build.log" 2>&1
  fi
  [ -s "$D/tmpl/job.json" ] || { echo "   !! template build failed"; tail -3 "$D/build.log" | sed 's/^/     /'; continue; }
  cd /home/ebrahim/Protenix || exit 1
  env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
      PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
      PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1 \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$D/tmpl/job.json" --dump_dir "$D/out" --seeds "$SEED" \
    --model.N_cycle $NCYCLE --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
    --use_template true > "$D/run.log" 2>&1
  cd "$REPO" || exit 1
  # assert: the template arrived, exactly NCYCLE recycles ran, NS structures written
  grep -oE 'Found [0-9]+ templates' "$D/run.log" | sort -u | sed 's/^/     /'
  grep -oE 'cycle=[0-9]+' "$D/run.log" | head -1 | sed 's/^/     /'
  grep -q 'PROTENIX_LOAD_TRUNK' "$D/run.log" && echo "     !! a trunk WAS loaded -- not a z_init run"
  N=$(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
  if [ "$N" -ne "$NS" ]; then
    echo "     !! $N/$NS poses -- not scoring"; grep -m2 -E "ERROR|Error" "$D/run.log" | cut -c1-140 | sed 's/^/        /'; continue
  fi
  $PY "$DC/row.py" --dir "$D" --iface "$IFACE" --seed "$SEED" --arm "$ARM" \
      --donor "$k" --ref "$REF" --scorer pxm >> "$CSV"
  grep "^${IFACE},${SEED},${ARM}," "$CSV" | awk -F, '{printf "     sample %s  dockq %-7s iptm %s\n",$5,$6,$7}'
done
echo "=== $(date +%H:%M:%S) done"
