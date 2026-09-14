#!/usr/bin/env bash
# COMMIT EACH POSE, ON A LEVEL PLAYING FIELD: one templated 11th recycle per pose, same start state.
#
# THE PROBLEM THIS ADDRESSES. 9x1h_EF seed 300014 produces five poses spanning DockQ 0.044-0.634 from
# ONE frozen pair rep, and ipTM spans only 0.117-0.151 across them -- confidence cannot tell a
# near-native pose from a wrong one, because all five come from the same trunk state and ipTM is a
# property of that state, not of the pose.
#
# THE IDEA. Give each pose its own trunk state, identically. Start from the SAME cached 10-recycle
# z/s, add exactly one recycle with that pose as a self-template, decode. Now the five arms differ in
# one thing only -- which pose was shown -- so their confidences are finally comparable. If they
# separate and track DockQ, that is a selector. If they all rise together, that is the attractor
# result confirmed within a single seed, which is worth knowing too.
#
# WHY THIS IS CHEAP. PROTENIX_LOAD_TRUNK starts the recycling carry-over from the dumped z/s instead
# of zeros, so N_cycle=1 continues the original run rather than redoing it: one recycle per arm
# instead of ten. The input features are still built normally from each arm's job json -- nothing is
# reconstructed by hand, which is what broke four earlier attempts at this.
#
# CONTROLS, both load-bearing:
#   arm "none"  -- an 11th recycle with NO template, isolating "one more recycle" from "a templated
#                  recycle". Without it, any change could just be the extra recycle.
#   MSA OFF + DROPOUT OFF -- PROTENIX_MSA_NROWS=1/topk pins the MSA to the query row only and
#                  mc_dropout_apply_rate 0 makes `random.random() < rate` always false. Otherwise
#                  recycle 11 resamples the MSA and rolls dropout independently in every arm and we
#                  would be measuring three things at once.
#
# FIVE DIFFUSION SAMPLES PER ARM, not one: the quantity of interest is whether the arm COMMITS, and
# commitment is the spread across samples. The trunk work dominates; extra samples are nearly free.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-5}
PDB=${PDB:-9x1h}; SEED=${SEED:-300014}; NS=${NS:-5}
DUMP=$REPO/artifacts/dump9x1h/reps/seed${SEED}_reps.npz
P=$REPO/artifacts/pxm2025
S=$REPO/artifacts/posecommit/${PDB}_${SEED}
mkdir -p "$S"
cd "$REPO" || exit 1

POSES=$(ls "$P/out/$PDB/$PDB/seed_$SEED/predictions/${PDB}_sample_"*.cif 2>/dev/null | sort)
NP=$(echo "$POSES" | grep -c .)
echo "=== $(date +%H:%M:%S) $PDB seed $SEED : $NP poses, dump $(basename "$DUMP")"
[ -s "$DUMP" ] || { echo "no dump"; exit 1; }

run_arm () {                     # run_arm <tag> <job.json> <extra-env>
  local tag=$1 job=$2 xenv=$3
  local D=$S/$tag
  mkdir -p "$D/out" "$D/pxm_json"
  echo "--- $(date +%H:%M:%S) arm $tag"
  cd /home/ebrahim/Protenix || exit 1
  env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
      PROTENIX_LOAD_TRUNK="$DUMP" \
      PROTENIX_MSA_NROWS=1 PROTENIX_MSA_STRATEGY=topk $xenv \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$job" --dump_dir "$D/out" \
    --seeds "$SEED" --model.N_cycle 1 \
    --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
    --mc_dropout_apply_rate 0.0 --use_template true > "$D/run.log" 2>&1
  cd "$REPO" || exit 1
  grep -q 'PROTENIX_LOAD_TRUNK' "$D/run.log" \
    && echo "    carry-over loaded OK" \
    || echo "    !! LOAD_TRUNK NEVER FIRED -- arm $tag started from zeros, not the cached state"
  grep -oE 'Found [0-9]+ templates' "$D/run.log" | sort | uniq -c | sed 's/^/    /'
  for f in $(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
    k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
    [ -s "$D/pxm_json/$k.json" ] || envs/pxmeter/bin/pxm -r "$P/cif/$PDB.cif" -m "$f" \
        -o "$D/pxm_json/$k.json" >/dev/null 2>&1
  done
  echo "    poses scored: $(ls "$D/pxm_json"/*.json 2>/dev/null | wc -l)"
}

# ---- control: an 11th recycle with no template at all -------------------------------------------
run_arm none "$P/jsons/${PDB}-final-updated.json" ""

# ---- one arm per pose ---------------------------------------------------------------------------
i=0
for pose in $POSES; do
  TID=$(printf 'q%02d%s' "$i" "$(echo $SEED | tail -c 2)")
  $PY scripts/build_selftmpl_general.py --pdb "$PDB" --out "$S/p$i" --poses "$pose" \
      --tids "$TID" >> "$S/build.log" 2>&1 \
    || { echo "    build failed for pose $i"; i=$((i+1)); continue; }
  run_arm "p$i" "$S/p$i/job.json" "PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1"
  i=$((i+1))
done

echo "=== $(date +%H:%M:%S) done"
$PY scripts/pose_commit_read.py --dir "$S" || true
