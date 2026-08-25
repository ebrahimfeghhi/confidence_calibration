#!/bin/bash
# Measure per-RECYCLE (trunk/pairformer) cost vs per-DIFFUSION-SAMPLE cost, on one GPU.
#
# METHOD. No code instrumentation -- vary one knob at a time and read the per-seed wall-clock that
# inference.py already logs ("Model forward time: N s"). With
#     T(c, s) = fixed + c * per_cycle + s * per_sample
# holding s fixed and varying c gives per_cycle from the slope; holding c fixed and varying s gives
# per_sample. Two points per line plus a third for a linearity check.
#
# WHY THIS MATTERS: an earlier claim that the diffusion decode dominates wall-clock was made with no
# measurement and later contradicted. This settles it with numbers.
#
# Uses the scheduler's exact worker environment (LAYERNORM_TYPE=openfold etc). A direct
# runner/inference.py call without it produced all-NaN coordinates once already.

set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-3}
PDB=${PDB:-8che}
JSON=/home/ebrahim/Protenix/foldbench_abag/jsons/${PDB}-final-updated.json
OUT=$REPO/artifacts/timing
mkdir -p $OUT

run() {   # $1=N_cycle  $2=N_sample
    local tag="c$1_s$2"
    rm -rf $OUT/$tag
    CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    $PY /home/ebrahim/Protenix/runner/inference.py \
        --model_name protenix-v2 --load_checkpoint_dir /home/ebrahim/checkpoint \
        --input_json_path $JSON --dump_dir $OUT/$tag --seeds 101 --use_template true \
        --model.N_cycle $1 --sample_diffusion.N_sample $2 --sample_diffusion.N_step 200 \
        > $OUT/$tag.log 2>&1
    local t=$(grep -o "Model forward time: [0-9.]*" $OUT/$tag.log | tail -1 | grep -o "[0-9.]*")
    local nan=$(grep -c 'nan nan nan' $(find $OUT/$tag -name '*_sample_0.cif' 2>/dev/null | head -1) 2>/dev/null || echo NA)
    echo "N_cycle=$1 N_sample=$2  forward=${t:-FAILED}s  nan_atoms=$nan"
}

echo "### timing on GPU $GPU, pdb $PDB, N_step 200 (started $(date +%H:%M))"
echo "--- vary N_cycle at N_sample=1 ---"
run 1 1
run 4 1
run 10 1
echo "--- vary N_sample at N_cycle=1 ---"
run 1 5
run 1 10
echo "### done $(date +%H:%M)"
