#!/bin/bash
# Dump the trunk contact map (distogram_head(z) -> contact_probs) for a list of 8che seeds, 4 GPUs.
#
# 7 seeds produce at least one acceptable pose (109 112 114 117 133 135 138) and 3 produce none
# (101 104 105), so the profiles can be compared between seeds that got a good pose and seeds that
# never did -- the question being whether the trunk's secondary epitope peak is what distinguishes
# them, or whether it is there either way and diffusion is the only difference.
#
# One worker loop per GPU with its own sublist, rather than xargs -P, so two jobs can never land on
# the same card (8che peaks around 15 GB but the sweep has shown timing skew is real).
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT=$REPO/artifacts/reps
mkdir -p "$OUT"
GPUS=(3 4 5 7)
SEEDS=(109 112 114 117 133 135 138 101 104 105)

worker() {
    local gpu=$1; shift
    for s in "$@"; do
        if [ -f "$OUT/8che_seed${s}.npz" ]; then echo "[gpu$gpu] seed $s already dumped"; continue; fi
        CUDA_VISIBLE_DEVICES=$gpu DUMP_REPS=1 DUMP_REPS_DIR=$OUT DUMP_REPS_TAG=8che_seed${s} \
        LAYERNORM_TYPE=openfold OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        $PY /home/ebrahim/Protenix/runner/inference.py \
            --model_name protenix-v2 --load_checkpoint_dir /home/ebrahim/checkpoint \
            --input_json_path /home/ebrahim/Protenix/foldbench_abag/jsons/8che-final-updated.json \
            --dump_dir "$OUT/run_seed${s}" --seeds $s --use_template true \
            --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
            --diffusion_chunk_size 5 > "$OUT/seed${s}.log" 2>&1
        if [ -f "$OUT/8che_seed${s}.npz" ]; then
            echo "[gpu$gpu] seed $s OK  $(du -h "$OUT/8che_seed${s}.npz" | cut -f1)"
        else
            echo "[gpu$gpu] *** seed $s FAILED -- $(grep -c 'DUMP_REPS failed' "$OUT/seed${s}.log") dump errors"
        fi
    done
}

# deal seeds round-robin across the 4 cards
for i in "${!GPUS[@]}"; do
    sub=()
    for j in "${!SEEDS[@]}"; do
        [ $((j % ${#GPUS[@]})) -eq "$i" ] && sub+=("${SEEDS[$j]}")
    done
    worker "${GPUS[$i]}" "${sub[@]}" &
done
wait
echo "### dumped $(ls $OUT/8che_seed*.npz 2>/dev/null | wc -l) npz files  $(date +%H:%M)"
