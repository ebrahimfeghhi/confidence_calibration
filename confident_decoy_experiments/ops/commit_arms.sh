#!/usr/bin/env bash
# Is the cycle-to-cycle epitope wander REAL, or just MSA/dropout resampling noise?
#
# One seed of 8slb showed the closest-residue set converging to 0.86 jaccard-to-final by cycle 4,
# regressing to 0.51 by cycle 6, then re-converging. That would mean there is no well-defined
# "epitope stabilisation loop" to push at -- but Protenix redraws the MSA subsample AND the MC
# dropout mask on EVERY recycle, so the wander may be pure resampling churn.
#
# ARM nodrop : full MSA + no dropout. data.msa.min_size.test=16384 makes sample_size == n (see
#              sample_indices: sample_size = randint(min(lower_bound,n), n+1)), and
#              mc_dropout_apply_rate=0 makes `random.random() < rate` always False.
# ARM stock  : stock inference, both stochastic. The comparison is the noise floor.
#
# Only the TRUNK matters here (the dumps are written inside the recycling loop), so decode is kept
# minimal: N_sample 1, N_step 20.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
JSON=/home/ebrahim/Protenix/foldbench_abag/jsons/8slb-final-updated.json
SEEDS=$(seq -s, 101 110)
cd /home/ebrahim/Protenix || exit 1
run(){
  arm=$1; shift
  echo "### ARM $arm $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold \
  DUMP_CYCLE_DIR=$REPO/artifacts/commit/$arm DUMP_CYCLE_TAG=8slb \
  DUMP_CYCLE_BINS="0:1,2" \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$JSON" --dump_dir $REPO/artifacts/commit/out_$arm \
    --seeds $SEEDS --sample_diffusion.N_sample 1 --sample_diffusion.N_step 20 \
    --use_template true "$@"
  echo "### ARM $arm DONE $(date +%H:%M) -> $(ls $REPO/artifacts/commit/$arm/*.npz 2>/dev/null | wc -l) dumps"
}
run nodrop2 --data.msa.min_size.test 16384 --mc_dropout_apply_rate 0.0
run stock2
echo "### ALL DONE $(date +%H:%M)"
