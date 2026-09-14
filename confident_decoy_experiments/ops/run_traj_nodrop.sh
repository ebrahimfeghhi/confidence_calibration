#!/usr/bin/env bash
# 7yxu trajectories with the stochasticity switched OFF: full MSA, no MC dropout.
#
# WHY: the decoded epitope on 7yxu swings wildly between recycles -- one seed hit jaccard 0.88 at
# recycle 7, 0.00 at recycle 8 and 0.75 at recycle 9. Two things could cause that: the trunk really
# relocating the epitope, or the per-recycle MSA subsample and dropout mask being redrawn. The
# distogram version of this comparison already showed nodrop and stock are indistinguishable, but the
# DECODED structure adds diffusion sampling on top, which the distogram never saw.
#
#   --data.msa.min_size.test 16384  makes sample_size == n, i.e. no MSA subsampling
#                                   (sample_indices: sample_size = randint(min(lower_bound,n), n+1))
#   --mc_dropout_apply_rate 0.0     makes `random.random() < rate` always false
#
# Same 8 seeds as the stock run so the two are directly comparable.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT=$REPO/artifacts/traj_nodrop/7yxu
SEEDS=$($PY -c "import json;print(','.join(str(s) for s in json.load(open('$REPO/artifacts/traj20_seeds.json'))['7yxu']['seeds']))")
cd /home/ebrahim/Protenix || exit 1
echo "### 7yxu nodrop trajectories on GPU5, seeds $SEEDS  $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=5 LAYERNORM_TYPE=openfold \
DUMP_CYCLE_DIR=$OUT DUMP_CYCLE_TAG=7yxu DUMP_CYCLE_DECODE=1 \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path /home/ebrahim/Protenix/foldbench_abag/jsons/7yxu-final-updated.json \
  --dump_dir $REPO/artifacts/traj_nodrop/out_7yxu \
  --seeds "$SEEDS" --sample_diffusion.N_sample 1 --sample_diffusion.N_step 200 \
  --data.msa.min_size.test 16384 --mc_dropout_apply_rate 0.0 --use_template true
echo "### DONE $(date +%H:%M) -> $(ls $OUT/*_coords.npz 2>/dev/null | wc -l) dumps"
