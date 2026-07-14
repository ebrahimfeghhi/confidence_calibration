#!/usr/bin/env bash
# The MSA ablation, with its CONTROL.
#
# THE DESIGN FLAW THIS FIXES. My first version ran base biohub/ESMFold2 WITH MSAs and compared it
# to ESMFold2-Fast (single-sequence). But those are DIFFERENT CHECKPOINTS (1.3 GB vs 721 MB), so
# that comparison varies the MSA *and* the model, and nothing could be attributed to the MSA.
#
# So we run BOTH arms on the SAME base checkpoint. The only variable is the MSA:
#
#   arm B  base ESMFold2 + Protenix's MSAs   -> out_dir esmfold2_msa
#   arm A' base ESMFold2, NO MSA (--no_msa)  -> out_dir esmfold2_nomsa    <-- THE CONTROL
#
# B - A' isolates the MSA. (ESMFold2-Fast single-sequence and Protenix stay in the comparison as
# reference points, but the causal claim rests on B vs A'.)
#
# Shared box: NEVER touch GPU 0 (shawn) or GPU 5 (kao). OMP_NUM_THREADS capped -- unset, torch
# grabs every core (that is how 8 procs became 220 of 288 and starved another user).
set -u
cd /home/ebrahim/tts_abag

NSEEDS="${NSEEDS:-12}"                    # 12 seeds x 5 samples = 60 poses/complex, both arms
MSA_GPUS="${MSA_GPUS:-1 2 3}"
NOMSA_GPUS="${NOMSA_GPUS:-4 6 7}"
CX="7txw 7zoz 8che 8fxv 8ido 8oni 8opr 8q6r 8qz3 8the 8ts0 8uky 8w84 8w90"
LOG=steer/experiments/logs/msa_ablation
mkdir -p "$LOG"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

launch() {                                 # $1=arm label  $2=out_dir  $3=extra flags  $4...=gpus
  local arm=$1 out=$2 extra=$3; shift 3
  local gpus="$*" ngpu=$#
  local i=0
  for g in $gpus; do
    local seeds
    seeds=$(python3 -c "print(' '.join(str(s) for s in range($NSEEDS) if s % $ngpu == $i))")
    [ -z "$seeds" ] && { i=$((i+1)); continue; }
    echo "  [$arm] GPU $g: seeds [$seeds]"
    nohup env CUDA_VISIBLE_DEVICES=$g ~/esm/.venv/bin/python run/esm/run_msa_ablation.py \
        --out_dir "$out" --complexes $CX --seeds $seeds $extra \
        > "$LOG/${arm}_gpu${g}.log" 2>&1 &
    i=$((i+1))
  done
}

echo "arm B  (base ESMFold2 + MSA)      -> GPUs $MSA_GPUS"
launch msa   /home/ebrahim/esm/foldbench_predictions/esmfold2_msa   ""          $MSA_GPUS
echo "arm A' (base ESMFold2, NO MSA)    -> GPUs $NOMSA_GPUS   [THE CONTROL]"
launch nomsa /home/ebrahim/esm/foldbench_predictions/esmfold2_nomsa "--no_msa"  $NOMSA_GPUS

sleep 25
echo
echo "=== MEASURED, 25s after launch (hard rule: <=128 procs, <=512 GB, <=64 cores) ==="
n=$(ps -u "$USER" --no-headers | wc -l)
m=$(ps -u "$USER" -o rss= | awk '{s+=$1} END{printf "%d", s/1048576}')
c=$(ps -u "$USER" -o pcpu= --no-headers | awk '{s+=$1} END{printf "%d", s/100}')
echo "  procs=$n/128   mem=${m}GB/512   cores=${c}/64   load=$(cut -d' ' -f1 /proc/loadavg)"
if [ "$n" -gt 128 ] || [ "$m" -gt 512 ] || [ "$c" -gt 64 ]; then
  echo "  *** OVER CEILING — KILLING NOW ***"; pkill -9 -f run_msa_ablation; exit 1
fi
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
