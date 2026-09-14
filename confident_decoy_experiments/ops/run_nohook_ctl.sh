#!/bin/bash
# CONTROL: is the trajectory run's failure caused by the dump hook itself?
#
# 7txw stock seed 135 folds correctly in all 5 samples (A|C DockQ 0.795-0.906). The trajectory run,
# which is the same command plus DUMP_CYCLE_DECODE=1, gives seed 135 A|C DockQ 0.028 -- and 0/8 of
# its seeds reach the epitope on an interface the stock pool solves at 0.906. DUMP_CYCLE_DECODE runs
# a full 200-step diffusion at EVERY recycle; each of those draws from the same global RNG stream as
# MSA subsampling and MC dropout, so by the last recycle the run has diverged from the stock
# trajectory with the same seed. run_traj.sh's header claims the hook "is an observer only" -- this
# tests that claim.
#
# Identical to run_traj.sh except the DUMP_CYCLE_* env vars are absent and N_sample is 5.
# Expect A|C ~0.8-0.9 on seed 135 if the hook is the cause; ~0.03 if seed 135 simply is not
# reproducible run-to-run (which would be a different, equally important finding).
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT=$REPO/artifacts/nohook_ctl
mkdir -p "$OUT"
cd /home/ebrahim/Protenix || exit 1
echo "### no-hook control, 7txw, 8 trajectory seeds, N_sample=5, GPU5  $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=5 LAYERNORM_TYPE=openfold \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path /home/ebrahim/Protenix/foldbench_abag/jsons/7txw-final-updated.json \
  --dump_dir "$OUT" \
  --seeds 101,102,107,112,114,135,141,145 \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true 2>&1 | tail -40
echo "### DONE $(date +%H:%M)  cifs: $(find "$OUT" -name '*_sample_*.cif' | wc -l)"
