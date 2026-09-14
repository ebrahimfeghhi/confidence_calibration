#!/usr/bin/env bash
# Per-recycle epitope trajectories on 20 interfaces (16 complexes), 8 diverse seeds each.
#
# WHY: on 8slb the decoded epitope changed almost every recycle -- consecutive recycles shared only
# 24-44% of their contacts, and every trajectory crossed between the correct and decoy basins several
# times. That was ONE interface. This repeats it on 20, stratified across confident decoys, solved
# interfaces and coverage failures, so we can ask whether hopping is a failure signature or just what
# recycling does everywhere.
#
# Seeds are chosen by farthest-point traversal on epitope jaccard (scripts/pick_diverse_seeds.py):
# 8 per complex, keeping the 5 most diverse AFTER the run, because Protenix is not bit-reproducible
# so a re-run need not land where the original did.
#
# DUMP_CYCLE_DECODE=1 runs a full 200-step diffusion at every recycle and saves the coordinates.
# It is an observer only -- nothing is fed back, so the trajectory matches a normal run.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
JOBS=$REPO/artifacts/traj20_seeds.json
LOCKS=$REPO/artifacts/traj_locks
GPU=$1
mkdir -p "$LOCKS" "$REPO/artifacts/traj_cycles"
cd /home/ebrahim/Protenix || exit 1
echo "### traj worker GPU$GPU $(date +%H:%M)"
while true; do
  claimed=""
  for pdb in $($PY -c "import json;print(' '.join(json.load(open('$JOBS'))))"); do
    d=$REPO/artifacts/traj_cycles/$pdb
    [ -d "$d" ] && [ "$(ls $d/*_coords.npz 2>/dev/null | wc -l)" -ge 80 ] && continue
    if mkdir "$LOCKS/$pdb" 2>/dev/null; then claimed="$pdb"; break; fi
  done
  [ -z "$claimed" ] && { echo "### GPU$GPU nothing left $(date +%H:%M)"; break; }
  seeds=$($PY -c "import json;print(','.join(str(s) for s in json.load(open('$JOBS'))['$claimed']['seeds']))")
  out=$REPO/artifacts/traj_cycles/$claimed
  mkdir -p "$out"
  echo "### GPU$GPU claimed $claimed seeds $seeds $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  DUMP_CYCLE_DIR=$out DUMP_CYCLE_TAG=$claimed DUMP_CYCLE_DECODE=1 DUMP_CYCLE_NSAMPLE=5 \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path /home/ebrahim/Protenix/foldbench_abag/jsons/${claimed}-final-updated.json \
    --dump_dir $REPO/artifacts/traj_cycles/out_$claimed \
    --seeds "$seeds" --sample_diffusion.N_sample 1 --sample_diffusion.N_step 200 \
    --use_template true >> $REPO/artifacts/traj_cycles/${claimed}.log 2>&1 \
    && echo "### GPU$GPU DONE $claimed $(ls $out/*_coords.npz 2>/dev/null | wc -l) dumps $(date +%H:%M)" \
    || echo "### GPU$GPU FAIL $claimed $(date +%H:%M)"
done
