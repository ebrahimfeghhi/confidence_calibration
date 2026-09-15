#!/usr/bin/env bash
# Re-fold a seed FROM SCRATCH with one of its own poses as the only template. One run per pose.
#
# HOW THIS DIFFERS FROM fold_and_commit.sh. The arms there resume a saved trunk and run ONE further
# recycle: the template gets a single pass to shape a pair rep that already has ten recycles of history
# behind it. Here there is no trunk to resume -- the fold starts from zeros and runs all 10 recycles WITH
# the template present, so the pose shapes the representation from the beginning.
#
# That difference tests two things at once:
#   - is one recycle too short for the template to take effect?
#   - was the 22% "revert to the seed's dominant basin" the trunk's memory, or the model's prior?
#     With no trunk carried over there is no memory to revert to.
#
# NO CONTROL ARM HERE, deliberately: the seed's original un-templated poses are the baseline and already
# exist. Note the caveat that comes with that -- this seed is NOT reproducible run-to-run (nine identical
# repeats of 9x1h seed 300014 spanned DockQ 0.004-0.651), so the baseline comes from a different
# invocation and a difference of that size proves nothing on its own.
#
# THE TEMPLATE FLAGS ARE LOAD-BEARING: templatesPath in the built job suppresses the real PDB search, so
# the self-pose is not competing with real hits; NODEDUP stops the pose being collapsed against the query
# (same sequence -- otherwise the run logs "Found 0 templates" and looks fine); MULTICHAIN_MASK=0 plus
# KEEP_FRAME preserve the inter-chain geometry, which is the entire content of a pose.
#
# Usage: GPU=5 POSES="0 1 2" bash selftmpl_refold.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}; POSES=${POSES:?}
PDB=${PDB:-9x1h}; IFACE=${IFACE:-9x1h_EF}; SEED=${SEED:-300014}
NCYCLE=${NCYCLE:-10}; NS=${NS:-5}
P=$REPO/artifacts/pxm2025
SRCDIR=$P/out/$PDB/$PDB/seed_$SEED/predictions
REF=$P/cif/$PDB.cif
OUT=$REPO/artifacts/commit11/selftmpl${NCYCLE}/${IFACE}_s${SEED}
CSV=$REPO/artifacts/commit11/results.csv
mkdir -p "$OUT"
cd "$REPO" || exit 1

for k in $POSES; do
  POSE=$SRCDIR/${PDB}_sample_${k}.cif
  [ -s "$POSE" ] || { echo "!! pose $k missing: $POSE"; continue; }
  D=$OUT/t$k
  ARM=t$k
  if grep -q "^${IFACE},${SEED},${ARM}," "$CSV" 2>/dev/null; then
    echo "=== $ARM already scored, skipping"; continue
  fi
  mkdir -p "$D/out"
  echo "=== $(date +%H:%M:%S) GPU$GPU pose $k (DockQ of the shown pose is the x-axis) -> $ARM"

  if [ ! -s "$D/tmpl/job.json" ]; then
    SELFTMPL_JOBS=$P/jsons $PY "$DC/build_selftmpl_general.py" --pdb "$PDB" --out "$D/tmpl" \
      --poses "$POSE" --tids "$(printf 'r%01d%02d' "$k" $((SEED % 100)))" >> "$D/build.log" 2>&1
  fi
  [ -s "$D/tmpl/job.json" ] || { echo "!! pose $k: template build FAILED"; tail -3 "$D/build.log" | sed 's/^/     /'; continue; }

  cd /home/ebrahim/Protenix || exit 1
  env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
      PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
      PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1 \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$D/tmpl/job.json" --dump_dir "$D/out" \
    --seeds "$SEED" --model.N_cycle $NCYCLE \
    --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
    --use_template true > "$D/run.log" 2>&1
  cd "$REPO" || exit 1

  # ASSERT THE MECHANISM FIRED. A run that silently found 0 templates, or produced no structures,
  # still yields clean-looking numbers -- both have happened here today.
  grep -oE 'Found [0-9]+ templates' "$D/run.log" | sort | uniq -c | sed 's/^/    /'
  grep -oE 'cycle=[0-9]+' "$D/run.log" | head -1 | sed 's/^/    /'
  N=$(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
  if [ "$N" -ne "$NS" ]; then
    echo "    !! produced $N/$NS poses -- NOT scoring, see $D/run.log"
    grep -m2 -E "ERROR|Error" "$D/run.log" | cut -c1-140 | sed 's/^/       /'
    continue
  fi
  $PY "$DC/row.py" --dir "$D" --iface "$IFACE" --seed "$SEED" --arm "$ARM" \
      --donor "$k" --ref "$REF" --scorer pxm >> "$CSV"
  echo "    $ARM: $N poses scored"
done
echo "=== $(date +%H:%M:%S) GPU$GPU done"
