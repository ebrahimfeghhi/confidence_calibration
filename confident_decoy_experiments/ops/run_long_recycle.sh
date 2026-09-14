#!/usr/bin/env bash
# Does 8a44 keep improving past the 10 recycles the model normally runs?
#
# Two of its seeds were still climbing toward the crystal epitope when the trunk stopped:
# seed 140 reached jaccard 0.30 at recycle 8 and 0.50 at recycle 9, seed 131 peaked 0.44 at recycle 8.
# 8a44 is classified a coverage failure (oracle DockQ 0.019 across 1078 seeds), so if the epitope
# keeps improving with more recycling that is a different failure mode entirely -- not "cannot reach
# the site" but "not given enough recycles to get there".
#
# CAVEAT: the model was trained at a fixed recycle count, so cycles 10-29 are out of distribution.
# Degradation would not prove the epitope is unreachable.
#
# Seeds 140,131 = the two that were climbing.  103,104 = flat-at-zero controls.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT=$REPO/artifacts/traj_long/8a44
cd /home/ebrahim/Protenix || exit 1
echo "### long-recycle 8a44 on GPU5, N_cycle=30 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=5 LAYERNORM_TYPE=openfold \
DUMP_CYCLE_DIR=$OUT DUMP_CYCLE_TAG=8a44long DUMP_CYCLE_DECODE=1 \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path /home/ebrahim/Protenix/foldbench_abag/jsons/8a44-final-updated.json \
  --dump_dir $REPO/artifacts/traj_long/out_8a44 \
  --seeds 140,131,103,104 --sample_diffusion.N_sample 1 --sample_diffusion.N_step 200 \
  --model.N_cycle 30 --use_template true
echo "### DONE $(date +%H:%M) -> $(ls $OUT/*_coords.npz 2>/dev/null | wc -l) dumps"
