#!/usr/bin/env bash
# Per seed: fold 10 recycles (dump trunk) -> ONE further recycle resuming that trunk with all 5 of its
# own poses as the sole template -> score both -> discard the trunk. Two stages, not seven.
#
# WHY ONE ARM AND NOT FIVE. The per-pose arms answered their question: showing the trunk a single pose
# reproduces it (r=0.985 at 10 recycles, r=0.92 at 1), so commitment works and there is nothing left to
# learn from five separate copies. Showing all five AT ONCE is the open question -- the model is not told
# which dock is right, it sees the seed's whole spread and must resolve it. On 8fas seed 29 that removed
# the seed's one bad pose (0.096) and returned all five at 0.966+, mean 0.801 -> 0.981, but that seed was
# already 4-of-5 good so it is the easy case; this sweep covers scattered seeds properly.
#
# THE 11TH RECYCLE IS A TRUE CONTINUATION: PROTENIX_LOAD_TRUNK initialises the recycling carry-over from
# the dumped z/s, so the single recycle starts where the fold stopped rather than from zeros. That line
# is ASSERTED -- an arm that silently starts from zeros still produces clean-looking numbers.
#
# Usage: GPU=5 PDB=8vqf IFACE=8vqf_AC SEEDS='1 2 3' bash commit_all5.sh
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}; PDB=${PDB:?}; IFACE=${IFACE:?}; SEEDS=${SEEDS:?}
NS=${NS:-5}; NCYCLE=${NCYCLE:-10}
DATASET=${DATASET:-pxm2025}
P=$REPO/artifacts/pxm2025
case "$DATASET" in
  pxm2025)   JOBS=$P/jsons; REF=$P/cif/$PDB.cif; SCORER=pxm ;;
  foldbench) JOBS=/home/ebrahim/Protenix/foldbench_abag/jsons
             REF=/home/ebrahim/FoldBench/examples/ground_truths/${PDB}-assembly1.cif; SCORER=ost ;;
  *) echo "!! unknown DATASET=$DATASET"; exit 1 ;;
esac
JOB=$JOBS/${PDB}-final-updated.json
CSV=$REPO/artifacts/commit11/results.csv
OUT=$REPO/artifacts/commit11/all5sweep/$IFACE
[ -s "$JOB" ] || { echo "!! no input json: $JOB"; exit 1; }
[ -s "$REF" ] || { echo "!! no reference: $REF"; exit 1; }
mkdir -p "$OUT"; cd "$REPO" || exit 1

for SEED in $SEEDS; do
  D=$OUT/s$SEED
  # Already has its all5 row? Nothing to do. Keyed on iface AND seed AND arm -- omitting the interface
  # is what once made every complex inherit another's completions and skip all its work.
  if grep -q "^${IFACE},${SEED},all5_k1," "$CSV" 2>/dev/null; then
    echo "=== seed $SEED already done, skipping"; continue
  fi
  mkdir -p "$D/fold/out"
  echo "=== $(date +%H:%M:%S) GPU$GPU $IFACE seed $SEED"

  # ---- stage 1: the fold. 10 recycles, dump the trunk, decode NS poses (these are the templates) ----
  if [ ! -s "$D/trunk.npz" ] || [ "$(find "$D/fold/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)" -ne "$NS" ]; then
    cd /home/ebrahim/Protenix || exit 1
    env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold PROTENIX_DUMP_TRUNK="$D/trunk.npz" \
    $PY runner/inference.py --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$JOB" --dump_dir "$D/fold/out" --seeds "$SEED" \
      --model.N_cycle $NCYCLE --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
      --use_template true > "$D/fold.log" 2>&1
    cd "$REPO" || exit 1
  fi
  # A TRUNK DUMP IS NOT PROOF OF A FOLD: it is written inside the forward pass, before any structure, so
  # a failed confidence head leaves a trunk and zero cifs. Count the poses.
  NP=$(find "$D/fold/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
  if [ "$NP" -ne "$NS" ] || [ ! -s "$D/trunk.npz" ]; then
    echo "    !! fold gave $NP/$NS poses, trunk=$([ -s "$D/trunk.npz" ] && echo yes || echo NO) -- skipping seed"
    grep -m2 -E "ERROR|Error" "$D/fold.log" 2>/dev/null | cut -c1-140 | sed 's/^/       /'
    rm -f "$D/trunk.npz"; continue
  fi
  grep -q "^${IFACE},${SEED},donor," "$CSV" 2>/dev/null || \
    $PY "$DC/row.py" --dir "$D/fold" --iface "$IFACE" --seed "$SEED" --arm donor \
        --donor -1 --ref "$REF" --scorer "$SCORER" >> "$CSV"

  # ---- stage 2: the 11th recycle, resuming the trunk, all 5 poses as the only template ----
  PS=$(find "$D/fold/out" -name '*_sample_*.cif' | head -1 | xargs dirname)
  GPU=$GPU SEED=$SEED PDB=$PDB IFACE=$IFACE DATASET=$DATASET NCYCLE=1 NS=$NS \
    POSE_SRC="$PS" LOAD_TRUNK="$D/trunk.npz" TAG="" ARM=all5_k1 \
    TIDPFX=f OUTDIR="$D" bash "$DC/allpose_refold.sh" 2>&1 | sed 's/^/    /'

  # the trunk is only needed WHILE this seed's arm runs; 89 interfaces x 50 seeds would otherwise be TBs
  rm -f "$D/trunk.npz"
  rm -rf "$D/fold/out" "$D/arm/out"   # scores are in the csv; the cifs are the bulk
done
echo "=== $(date +%H:%M:%S) GPU$GPU $IFACE: seeds done"
