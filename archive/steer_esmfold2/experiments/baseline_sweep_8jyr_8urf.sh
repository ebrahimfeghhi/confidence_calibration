#!/bin/bash
# Baseline (no steering) sweep: 1024 seeds each of 8jyr then 8urf, standard TTS
# recipe (10% LM mask, 30% dropout), scored for ab-ag DockQ/ipTM. Tests whether
# either rescue found by the repel campaign is reachable by scale alone.
set -e
cd /home/ebrahim/tts_abag
PYTHON=/home/ebrahim/esm/.venv/bin/python

RUN=baseline_8jyr_8urf_1024
PRED_DIR="$HOME/esm/foldbench_predictions/$RUN"
OUT_CSV="results/csv_files/dockq_ost_${RUN}_all_samples.csv"
mkdir -p results/csv_files

score_loop() {
    while true; do
        $PYTHON eval/eval_ost_foldbench.py --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV" \
            >> "steer/experiments/baseline_sweep_score.log" 2>&1
        sleep 600
    done
}
score_loop &
SCORE_PID=$!
trap 'kill $SCORE_PID 2>/dev/null' EXIT

for target in 8jyr 8urf; do
    echo "===== [$(date)] generating $target (1024 seeds) ====="
    $PYTHON run/esm/run_foldbench_abag.py --run_name "$RUN" --complexes "$target" \
        --seeds $(seq 0 1023) --lm_mask_pct 0.1 --lm_dropout 0.3 \
        --num_diffusion_samples 5 --chunk_size 32

    echo "===== [$(date)] $target generation done, final score pass ====="
    $PYTHON eval/eval_ost_foldbench.py --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV"

    echo "===== [$(date)] $target scored, reclaiming disk (raw predictions no longer needed) ====="
    rm -rf "${PRED_DIR:?}/${target}"
done

kill $SCORE_PID 2>/dev/null
trap - EXIT
echo "===== [$(date)] ALL DONE: baseline_8jyr_8urf_1024 ====="
