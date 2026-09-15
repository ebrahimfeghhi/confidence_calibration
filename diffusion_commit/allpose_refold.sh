#!/usr/bin/env bash
# Re-fold a seed with ALL FIVE of its own poses supplied together as templates. One run per seed.
#
# Contrast with selftmpl_refold.sh, which gives the model ONE pose and gets a faithful copy of it back
# (corr(shown, output) = +0.985). Here the model sees all five at once, so it is not being told which
# dock is right -- it is being shown the seed's whole spread and left to choose. MAX_TEMPLATES=5 is the
# cap that binds at inference; leaving it at the trained default of 4 would silently drop a pose.
#
# NODEDUP is essential: the five poses share the query's sequence, so by default they collapse to one
# (or none) and the run logs "Found 0 templates" while looking perfectly healthy. MULTICHAIN_MASK=0 and
# KEEP_FRAME preserve the inter-chain geometry, which is the only thing a pose actually encodes.
#
# Usage: GPU=5 SEED=300014 bash allpose_refold.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}; SEED=${SEED:?}
PDB=${PDB:-9x1h}; IFACE=${IFACE:-9x1h_EF}; NCYCLE=${NCYCLE:-10}; NS=${NS:-5}
P=$REPO/artifacts/pxm2025
# DATASET picks inputs, scoring reference and scorer TOGETHER -- mixing pools makes numbers
# incomparable, so it is one switch rather than three.
DATASET=${DATASET:-pxm2025}
case "$DATASET" in
  pxm2025)   JOBS=$P/jsons; REF=$P/cif/$PDB.cif; SCORER=pxm ;;
  foldbench) JOBS=/home/ebrahim/Protenix/foldbench_abag/jsons
             REF=/home/ebrahim/FoldBench/examples/ground_truths/${PDB}-assembly1.cif; SCORER=ost ;;
  *) echo "!! unknown DATASET=$DATASET (want pxm2025 or foldbench)"; exit 1 ;;
esac
# POSE_SEED lets the TEMPLATES come from one seed while inference runs under another, which separates
# "these five poses are informative" from "this seed happened to work". POSE_SRC overrides the directory
# outright, for chaining a second generation off a previous run's output.
POSE_SEED=${POSE_SEED:-$SEED}
POSE_SRC=${POSE_SRC:-$P/out/$PDB/$PDB/seed_$POSE_SEED/predictions}
SRC=$POSE_SRC
[ -s "$REF" ] || { echo "!! no reference: $REF"; exit 1; }
# OUTDIR lets a caller place this arm inside its own per-seed directory, so a sweep can clean up after
# itself instead of scattering output across allpose1/ and allpose10/.
D=${OUTDIR:+$OUTDIR/arm}
D=${D:-$REPO/artifacts/commit11/allpose${NCYCLE}/${IFACE}_s${SEED}${TAG:+_$TAG}}
CSV=$REPO/artifacts/commit11/results.csv
ARM=${ARM:-all5}
mkdir -p "$D/out"; cd "$REPO" || exit 1

POSES=""; TIDS=""
for k in 0 1 2 3 4; do
  f=$(ls "$SRC"/*_sample_${k}.cif 2>/dev/null | head -1)
  [ -s "$f" ] || { echo "!! missing pose $k"; exit 1; }
  POSES="$POSES $f"; TIDS="$TIDS $(printf '%s%01d%02d' "${TIDPFX:-a}" "$k" $((SEED % 100)))"
done
echo "=== $(date +%H:%M:%S) GPU$GPU seed $SEED: all 5 poses as templates, $NCYCLE recycles"

if [ ! -s "$D/tmpl/job.json" ]; then
  SELFTMPL_JOBS=$JOBS $PY "$DC/build_selftmpl_general.py" --pdb "$PDB" --out "$D/tmpl" \
    --poses $POSES --tids $TIDS >> "$D/build.log" 2>&1
fi
[ -s "$D/tmpl/job.json" ] || { echo "!! template build FAILED"; tail -4 "$D/build.log" | sed 's/^/   /'; exit 1; }

cd /home/ebrahim/Protenix || exit 1
# LOAD_TRUNK resumes a dumped trunk instead of starting from zeros, so with NCYCLE=1 this is the
# "11th recycle" arm -- but with all five poses as templates rather than one.
env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
    ${LOAD_TRUNK:+PROTENIX_LOAD_TRUNK=$LOAD_TRUNK} \
    PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
    PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=5 PROTENIX_MAX_TEMPLATE_CANDIDATES=20 \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$D/tmpl/job.json" --dump_dir "$D/out" \
  --seeds "$SEED" --model.N_cycle $NCYCLE \
  --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
  --use_template true > "$D/run.log" 2>&1
cd "$REPO" || exit 1

# ASSERT: five templates found, ten recycles, five structures. All three have failed silently before.
[ -n "${LOAD_TRUNK:-}" ] && { grep -q 'PROTENIX_LOAD_TRUNK' "$D/run.log" || echo "    !! LOAD_TRUNK NEVER FIRED"; }
grep -oE 'Found [0-9]+ templates' "$D/run.log" | sort | uniq -c | sed 's/^/    /'
grep -oE 'cycle=[0-9]+' "$D/run.log" | head -1 | sed 's/^/    /'
N=$(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
if [ "$N" -ne "$NS" ]; then
  echo "    !! produced $N/$NS poses -- NOT scoring, see $D/run.log"
  grep -m2 -E "ERROR|Error" "$D/run.log" | cut -c1-140 | sed 's/^/       /'; exit 1
fi
grep -q "^${IFACE},${SEED},${ARM}," "$CSV" 2>/dev/null || \
  $PY "$DC/row.py" --dir "$D" --iface "$IFACE" --seed "$SEED" --arm "$ARM" \
      --donor -1 --ref "$REF" --scorer "$SCORER" >> "$CSV"
echo "    $ARM: $N poses scored"
echo "=== $(date +%H:%M:%S) done"
