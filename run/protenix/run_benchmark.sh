#!/bin/bash
# Runs 113-structure ab-ag benchmark across 4 GPUs, 10 seeds, with templates enabled.
# Usage: bash run_benchmark.sh [dump_dir]
# To stop early: kill $(cat /tmp/benchmark_pids.txt)
set -uo pipefail

# --- resolve repo root and load centralized paths from paths.py ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ ! -f "$REPO_ROOT/paths.py" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
eval "$(python3 "$REPO_ROOT/paths.py" --sh)"

# Must run from the Protenix repo — runner/inference.py is part of that package.
cd "$PROTENIX_REPO"

DUMP_DIR="${1:-$PROTENIX_PREDICTIONS}"
CHECKPOINT="$PROTENIX_CKPT"
PYTHON="$PROTENIX_PYTHON"
GPUS=(0 1 3 4)
SEEDS=$(seq -s, 101 110)

mapfile -t ALL_JSONS < <(ls "$JSON_DIR"/*-final-updated.json | sort)
N_TOTAL=${#ALL_JSONS[@]}
N_GPUS=${#GPUS[@]}
mkdir -p "$DUMP_DIR"

echo "=== Benchmark: $N_TOTAL structures x 10 seeds across $N_GPUS GPUs ==="
echo "=== Dump dir: $DUMP_DIR ==="

run_gpu() {
    local GPU=$1; shift
    local LOG="$DUMP_DIR/gpu${GPU}.log"
    echo "[GPU $GPU] Starting $(date)" | tee -a "$LOG"

    for json in "$@"; do
        pdb_id=$(basename "$json" | sed 's/-final-updated\.json//')
        done_marker="$DUMP_DIR/$pdb_id/seed_110/predictions/${pdb_id}_summary_confidence_sample_0.json"
        if [ -f "$done_marker" ]; then
            echo "[GPU $GPU] SKIP $pdb_id (already done)" | tee -a "$LOG"
            continue
        fi
        echo "[GPU $GPU] START $pdb_id $(date)" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
            $PYTHON runner/inference.py \
            --model_name protenix-v2 \
            --load_checkpoint_dir "$CHECKPOINT" \
            --input_json_path "$json" \
            --dump_dir "$DUMP_DIR" \
            --seeds $SEEDS \
            --sample_diffusion.N_sample 5 \
            --sample_diffusion.N_step 200 \
            --use_template true >> "$LOG" 2>&1 \
            && echo "[GPU $GPU] DONE  $pdb_id $(date)" | tee -a "$LOG" \
            || echo "[GPU $GPU] FAIL  $pdb_id $(date)" | tee -a "$LOG"
    done
    echo "[GPU $GPU] All done $(date)" | tee -a "$LOG"
}

export -f run_gpu
export DUMP_DIR CHECKPOINT PYTHON SEEDS

PIDS=()
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    gpu_jsons=()
    for ((j=i; j<N_TOTAL; j+=N_GPUS)); do
        gpu_jsons+=("${ALL_JSONS[$j]}")
    done
    echo "GPU $GPU: ${#gpu_jsons[@]} structures → ${gpu_jsons[*]##*/}" | sed 's/-final-updated\.json//g'
    bash -c 'run_gpu "$@"' _ "$GPU" "${gpu_jsons[@]}" &
    PIDS+=($!)
done

printf '%s\n' "${PIDS[@]}" > /tmp/benchmark_pids.txt
echo ""
echo "Launched. PIDs saved to /tmp/benchmark_pids.txt"
echo "Progress logs: $DUMP_DIR/gpu{0..4}.log"
echo "Stop early:    kill \$(cat /tmp/benchmark_pids.txt)"
echo ""

wait "${PIDS[@]}"
echo "=== All GPUs finished ==="
