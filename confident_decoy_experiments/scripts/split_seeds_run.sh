#!/bin/bash
# Split one pdb's REMAINING seeds in [LO,HI] across several GPUs.
# A job cannot be split within a seed, but its seed list can be partitioned; with seed-level resume
# this is safe and turns a long serial run into a parallel one (7zau: 11 h -> ~2 h).
# usage: split_seeds_run.sh <pdb> <json> <name> <lo> <hi> <gpu> [<gpu> ...]
set -u
EXP=/home/ebrahim/confident_decoy_experiments
PDB=$1; JSON=$2; NAME=$3; LO=$4; HI=$5; shift 5; GPUS=("$@")
# OUT_DIR lets this serve any arm, not just native50 (dockqdonor writes elsewhere).
OUT=${OUT_DIR:-$EXP/artifacts/native_run_out/native50}
TODO=$(python3 -c "
import os
D='$OUT/$NAME'
print(','.join(str(s) for s in range($LO, $HI+1)
      if not os.path.exists(f'{D}/seed_{s}/predictions/${NAME}_summary_confidence_sample_4.json')))")
if [ -z "$TODO" ]; then echo "$PDB: nothing to do in [$LO,$HI]"; exit 0; fi
IFS=',' read -ra SEEDS <<< "$TODO"
N=${#SEEDS[@]}; G=${#GPUS[@]}
echo "$PDB: $N seeds to run in [$LO,$HI] across GPUs ${GPUS[*]}"
for i in "${!GPUS[@]}"; do
  part=""
  for ((j=i; j<N; j+=G)); do part="$part,${SEEDS[$j]}"; done
  part=${part#,}
  [ -z "$part" ] && continue
  cd /home/ebrahim/Protenix
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} LAYERNORM_TYPE=openfold OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  nohup .venv/bin/python runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint --input_json_path "$JSON" \
    --dump_dir "$OUT" --seeds "$part" \
    --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 --use_template true \
    > $EXP/artifacts/native_run_logs/${NAME}_ext_gpu${GPUS[$i]}.log 2>&1 &
done
wait
