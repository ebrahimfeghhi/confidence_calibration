#!/bin/bash
# One run per complex: N_cycle=10, N_sample=5, diffusion_chunk_size=8 so all 5 samples decode in a
# SINGLE batched pass (the default of 4 splits them into 4+1 sequential chunks).
#
# PROTENIX_TIMING=1 prints Protenix's own time_tracker split (pairformer vs diffusion) and adds
# torch.cuda.synchronize() at each phase boundary -- without the sync the trunk's asynchronously
# launched kernels get charged to whatever phase runs next, under-counting the trunk and
# over-counting diffusion.
set -uo pipefail
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT=/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/timing
mkdir -p "$OUT"
GPU=${GPU:-3}
for PDB in 8dtn 8che 8udz; do
    tag="ph_${PDB}"
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
    f=$(find "$OUT/$tag" -name '*_sample_0.cif' 2>/dev/null | head -1)
    if [ -n "$f" ]; then nan=$(grep -c 'nan nan nan' "$f"); else nan=NOFILE; fi
    echo "$PDB :: ${line:-NO TIMING LINE}  nan=$nan"
done
echo "### done $(date +%H:%M)"
