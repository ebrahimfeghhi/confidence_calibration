#!/usr/bin/env bash
# How much does the ACTUAL epitope change across recycles?
#
# The distogram turned out to be a broad, hedged readout: AUC 0.90 for the epitope region but only
# +0.06 correlation with where the decoded pose ends up. So instead of inferring the epitope from the
# distogram, this decodes the structure at EVERY recycle (DUMP_CYCLE_DECODE=1, observer only -- no
# feedback, trajectory unchanged) and saves the coordinates, plus the cross-chain distogram.
#
# 10 stock seeds of 8slb, split across the three GPUs. Full 200 diffusion steps, N_sample 1.
# Cost is one full diffusion rollout per recycle, i.e. ~10x a normal run.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
JSON=/home/ebrahim/Protenix/foldbench_abag/jsons/8slb-final-updated.json
GPU=$1; SEEDS=$2; TAG=g$GPU
OUT=$REPO/artifacts/decode_cycles/$TAG
mkdir -p "$OUT"
echo "$SEEDS" > "$OUT/seed_order.txt"     # pass NN in the filenames == NN-th seed in this list
cd /home/ebrahim/Protenix || exit 1
echo "### GPU$GPU seeds $SEEDS  $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
DUMP_CYCLE_DIR=$OUT DUMP_CYCLE_TAG=8slb DUMP_CYCLE_BINS="0:1,2" DUMP_CYCLE_DECODE=1 \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$JSON" --dump_dir $REPO/artifacts/decode_cycles/out_$TAG \
  --seeds "$SEEDS" --sample_diffusion.N_sample 1 --sample_diffusion.N_step 200 \
  --use_template true
echo "### GPU$GPU DONE $(date +%H:%M) -> $(ls $OUT/*_coords.npz 2>/dev/null | wc -l) coord dumps"
