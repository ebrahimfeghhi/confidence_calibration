#!/bin/bash
# MASKED + no-MSA controls for the unmask factorial (cells G, H).
# Run from the STOCK tree /home/ebrahim/Protenix, so multichain_mask is the normal intra-chain mask.
# Proof of condition: the log must contain NO "UNMASKED TEMPLATES ACTIVE" line and every module path
# must be /home/ebrahim/Protenix/... (not Protenix_unmask). Same donor 137_3, same 10 seeds.
set -u
A=/home/ebrahim/confident_decoy_experiments/artifacts
J=$A/sweep_tmpl/unmask8che_nomsa/8che_unmask_nomsa_137_3.json
S=137,108,130,250,332,832,1036,1040,1074,1076
PY=/home/ebrahim/Protenix/.venv/bin/python
mkdir -p $A/unmaskfac_logs
run () {   # $1 cell   $2 extra flags
  local c=$1; shift
  mkdir -p $A/sweep_out/unmaskfac_$c
  echo "[$(date +%H:%M)] cell $c starting (masked, no MSA) flags: $*"
  cd /home/ebrahim/Protenix
  CUDA_VISIBLE_DEVICES=7 LAYERNORM_TYPE=openfold OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path $J --dump_dir $A/sweep_out/unmaskfac_$c --seeds $S \
    --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 --use_template true "$@" \
    > $A/unmaskfac_logs/$c.log 2>&1
  echo "[$(date +%H:%M)] cell $c done; unmask lines (MUST be 0): $(grep -c 'UNMASKED TEMPLATES ACTIVE' $A/unmaskfac_logs/$c.log); N_msa: $(grep -o 'N_msa [0-9]*' $A/unmaskfac_logs/$c.log | sort -u | tr '\n' ' '); templates: $(grep -c 'Found 1 templates' $A/unmaskfac_logs/$c.log)"
}
run G
run H --mc_dropout_apply_rate 0
echo "[$(date +%H:%M)] MASKED NO-MSA CONTROLS DONE"
