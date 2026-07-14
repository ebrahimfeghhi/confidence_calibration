#!/usr/bin/env bash
# Launch the MSA ablation across GPUs. Splits the SEED axis (not the complex axis) so every GPU
# loads the model once and every complex gets the same seed coverage even if a GPU dies.
#
# Base ESMFold2 (msa_encoder.enabled = TRUE) + Protenix's own MSAs, on the 14 complexes of the
# selection-critical band. Settles whether Protenix's selector advantage is really just MSAs.
#
# Shared box: NEVER touch GPU 0 (shawn) or GPU 5 (kao). OMP_NUM_THREADS capped -- torch grabs
# every core otherwise (that is how 8 procs became 220 of 288 cores and starved another user).
set -u
cd /home/ebrahim/tts_abag

GPUS="${GPUS:-1 2 3 4 6 7}"
SEEDS_PER_GPU="${SEEDS_PER_GPU:-}"        # computed below
NSEEDS="${NSEEDS:-25}"
OUT="${OUT:-/home/ebrahim/esm/foldbench_predictions/esmfold2_msa}"
LOG=steer/experiments/logs/msa_ablation
mkdir -p "$LOG"

# the 14 complexes of the selection-critical band (pools 10-50% acceptable)
CX="7txw 7zoz 8che 8fxv 8ido 8oni 8opr 8q6r 8qz3 8the 8ts0 8uky 8w84 8w90"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# refuse to start on a GPU somebody else is on
for g in $GPUS; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v G="$g" '$1==G{print $2}')
  if [ "${used:-0}" -gt 500 ]; then
    echo "!!! GPU $g is in use (${used} MiB). Refusing to launch." ; exit 1
  fi
done

ngpu=$(echo $GPUS | wc -w)
i=0
for g in $GPUS; do
  # round-robin the seeds so each GPU gets an interleaved share
  seeds=$(seq 0 $((NSEEDS-1)) | awk -v k=$i -v n=$ngpu 'NR%n==k+1-1 || (k==0 && NR%n==0)' | tr '\n' ' ')
  seeds=$(python3 -c "
import sys
n,k,ng=$NSEEDS,$i,$ngpu
print(' '.join(str(s) for s in range(n) if s % ng == k))")
  [ -z "$seeds" ] && { i=$((i+1)); continue; }
  echo "GPU $g: seeds [$seeds]"
  nohup env CUDA_VISIBLE_DEVICES=$g ~/esm/.venv/bin/python run/esm/run_msa_ablation.py \
      --out_dir "$OUT" --complexes $CX --seeds $seeds \
      > "$LOG/gpu${g}.log" 2>&1 &
  i=$((i+1))
done

sleep 20
echo
echo "=== MEASURED, 20s after launch (hard rule: <=128 procs, <=512 GB, <=64 cores) ==="
n=$(ps -u "$USER" --no-headers | wc -l)
m=$(ps -u "$USER" -o rss= | awk '{s+=$1} END{printf "%d", s/1048576}')
c=$(ps -u "$USER" -o pcpu= --no-headers | awk '{s+=$1} END{printf "%d", s/100}')
echo "  procs=$n/128   mem=${m}GB/512   cores=${c}/64   load=$(cut -d' ' -f1 /proc/loadavg)"
if [ "$n" -gt 128 ] || [ "$m" -gt 512 ] || [ "$c" -gt 64 ]; then
  echo "  *** OVER CEILING — KILLING NOW ***"
  pkill -9 -f run_msa_ablation
  exit 1
fi
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
