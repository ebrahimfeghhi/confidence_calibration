#!/bin/bash
# GPU 6 supplements the baseline sweep by taking over GPU 1's untouched shard (GPU 1 never
# freed). Normal complexes -> seeds 50-149 (100 new mask0.1 seeds), monsters -> 50-69 (cap 20).
# Writes into esmfold2fast_tts with skip-existing, so it's safe even if GPU 1 ever wakes up.
set -u
cd /home/ebrahim/tts_abag
PY=/home/ebrahim/esm/.venv/bin/python
NORMAL=$(cat steer/experiments/prog_state/normal_1.txt)
MONSTER=$(cat steer/experiments/prog_state/monster_1.txt)
LOG=steer/experiments/gpu6_baseline.log
echo "[$(date '+%F %T')] GPU6 baseline: $(echo $NORMAL|wc -w) normal + $(echo $MONSTER|wc -w) monster (GPU 1's shard)" | tee "$LOG"
CUDA_VISIBLE_DEVICES=6 $PY run/esm/run_foldbench_abag.py --run_name esmfold2fast_tts \
  --complexes $NORMAL --seeds $(seq 50 149) --lm_mask_pct 0.1 --lm_dropout 0.3 \
  --num_diffusion_samples 5 --chunk_size 32 --fused --no_config_log >>"$LOG" 2>&1
echo "[$(date '+%F %T')] GPU6 normal done, starting monsters (capped 20)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=6 $PY run/esm/run_foldbench_abag.py --run_name esmfold2fast_tts \
  --complexes $MONSTER --seeds $(seq 50 69) --lm_mask_pct 0.1 --lm_dropout 0.3 \
  --num_diffusion_samples 5 --chunk_size 32 --fused --no_config_log >>"$LOG" 2>&1
echo "[$(date '+%F %T')] GPU6 baseline DONE" | tee -a "$LOG"
