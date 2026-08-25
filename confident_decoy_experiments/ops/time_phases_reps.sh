#!/bin/bash
# Repeat the phase timing to measure run-to-run VARIANCE. Same seed, same config, same GPU -- so any
# spread here is wall-clock noise (GPU clocks, contention with the sweep on 5/7/4, filesystem), not
# model variability. Reps 2 and 3; rep 1 is the earlier ops/time_phases.sh run.
set -uo pipefail
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT=/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/timing
mkdir -p "$OUT"
GPU=${GPU:-3}
for REP in 2 3; do
    for PDB in 8dtn 8che 8udz; do
        tag="ph_${PDB}_r${REP}"
        rm -rf "$OUT/$tag"
        CUDA_VISIBLE_DEVICES=$GPU PROTENIX_TIMING=1 LAYERNORM_TYPE=openfold OMP_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        $PY /home/ebrahim/Protenix/runner/inference.py \
            --model_name protenix-v2 --load_checkpoint_dir /home/ebrahim/checkpoint \
            --input_json_path "/home/ebrahim/Protenix/foldbench_abag/jsons/${PDB}-final-updated.json" \
            --dump_dir "$OUT/$tag" --seeds 101 --use_template true \
            --model.N_cycle 10 --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
            --diffusion_chunk_size 8 > "$OUT/$tag.log" 2>&1
        line=$(grep -o "\[timing\].*" "$OUT/$tag.log" | tail -1)
        echo "rep$REP $PDB :: ${line:-NO TIMING LINE}"
    done
done
echo "### reps done $(date +%H:%M)"
