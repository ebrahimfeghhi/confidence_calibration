#!/usr/bin/env bash
# 9d41_AB, donor seed 300005: template on EACH of its five poses individually.
#
# THE QUESTION. Templating all five of that seed's poses together gave outputs 0.090 0.101 0.072
# 0.039 0.114 -- the donor set contained a 0.280 pose and the ensemble did not reproduce it. Across
# the four harm interfaces, donor sets whose best member was >=0.15 were reproduced-or-beaten in only
# 15 of 69 cases, so the ensemble appears to average toward the majority, which in a 1-good/4-bad set
# is wrong. Templating each pose ALONE tests that directly.
#
# CONTROL: all five runs use seed 300030 -- exactly the seed the ensemble run used (300005 + 25). The
# MSA draw and diffusion noise are therefore held fixed and the ONLY variable is which template is
# supplied. Using fresh seeds would reintroduce the trunk lottery, which on this project swings DockQ
# by 0.25 on identical inputs.
#
# STOCK CONFIG (MSA subsampling on, mc_dropout at its 0.4 default) to match the arm it is being
# compared against. MAX_TEMPLATES=1 with NODEDUP still set, because the self-pose would otherwise
# dedup against the query by mmcif sequence and log "Found 1 templates" for the wrong reason.
# MULTICHAIN_MASK=0 + TMPL_KEEP_FRAME=1 keep the real inter-chain geometry.
#
# PREDICTION: the run built from the 0.280 pose returns ~0.25-0.28 and the rest stay low. If the
# 0.280 also collapses, that pose simply is not reproducible and the ensemble is not at fault.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-4}
PDB=9d41; IFACE=9d41_AB; DSEED=300005; RSEED=300030
P=$REPO/artifacts/pxm2025
S=$REPO/artifacts/single9d41
mkdir -p "$S"
cd "$REPO" || exit 1

echo "=== $(date +%H:%M:%S) donor seed $DSEED, five poses templated individually, run seed $RSEED"
for k in 0 1 2 3 4; do
  POSE="$P/out/$PDB/$PDB/seed_$DSEED/predictions/${PDB}_sample_${k}.cif"
  [ -f "$POSE" ] || { echo "   missing $POSE"; continue; }
  D=$S/p$k
  mkdir -p "$D/out" "$D/pxm_json"
  echo "--- pose $k"
  $PY scripts/build_selftmpl_general.py --pdb "$PDB" --out "$D" --poses "$POSE" \
      --tids "y00$k" --no-cache-write 2>&1 | sed 's/^/     /'
  cd /home/ebrahim/Protenix || exit 1
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
  PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1 \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$D/job.json" --dump_dir "$D/out" \
    --seeds "$RSEED" --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
    --use_template true > "$D/run.log" 2>&1
  cd "$REPO" || exit 1
  grep -q 'Found 0 templates' "$D/run.log" && echo "     !! ran UNTEMPLATED"
  for f in $(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
    s=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
    [ -s "$D/pxm_json/$s.json" ] || envs/pxmeter/bin/pxm -r "$P/cif/$PDB.cif" -m "$f" \
        -o "$D/pxm_json/$s.json" >/dev/null 2>&1
  done
  rm -f "/home/ebrahim/mmcif/y00$k.cif"
done
echo "=== $(date +%H:%M:%S) done"
