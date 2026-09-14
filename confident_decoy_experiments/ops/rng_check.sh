#!/usr/bin/env bash
# Does the same seed give the same MSA draw and the same dropout decisions across ARMS?
#
# WHY IT MATTERS. In commit11 the six arms differ in ipTM by as little as 0.003. If the arms also
# differ in which MSA rows they saw, that spread is not attributable to the template and the whole
# comparison is worthless. seed_everything() seeds random/numpy/torch per run, so identical runs
# agree -- but arms are NOT identical upstream: the control featurises 4 templates and the p-arms 1,
# and any RNG consumed before the MSA draw shifts the stream.
#
# FOUR RUNS, all seed 300000, N_cycle=1, loading the same trunk state:
#   A1, A2  the control arm, twice      -> is a single configuration even reproducible?
#   B       the p0 arm (1 self-template) -> does a different template count shift the draw?
#   C       the control with dropout ON  -> are the dropout decisions themselves reproducible?
# Diffusion is cut to N_sample=1, N_step=5: we are reading log lines, not scoring structures.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
PDB=9h1f; SEED=300000
P=$REPO/artifacts/pxm2025
D=$REPO/artifacts/commit11/9h1f_AB_s300000
O=$REPO/artifacts/rngcheck
mkdir -p "$O"
cd "$REPO" || exit 1
[ -s "$D/trunk.npz" ] || { echo "need $D/trunk.npz from the commit11 run"; exit 1; }

go () {                    # go <tag> <job.json> <extra-env> <dropout-rate>
  local tag=$1 job=$2 xenv=$3 dr=$4
  cd /home/ebrahim/Protenix || exit 1
  env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
      PROTENIX_LOAD_TRUNK="$D/trunk.npz" PROTENIX_LOG_MSA=1 PROTENIX_LOG_MC=1 $xenv \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$job" --dump_dir "$O/$tag" \
    --seeds "$SEED" --model.N_cycle "${NC:-1}" \
    --sample_diffusion.N_sample 1 --sample_diffusion.N_step 5 \
    --mc_dropout_apply_rate "$dr" --use_template true > "$O/$tag.log" 2>&1
  cd "$REPO" || exit 1
  echo "--- $tag"
  grep -oE "MSA_DRAW n=[0-9]+ sample_size=[0-9]+.*" "$O/$tag.log" | head -2 | sed 's/^/    /'
  echo -n "    dropout decisions: "; grep -oE "MC_DROPOUT decision=\w+" "$O/$tag.log" | tr '\n' ' '; echo
  grep -oE "Found [0-9]+ templates" "$O/$tag.log" | head -1 | sed 's/^/    /'
}

TENV="PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1"
echo "=== control, run 1 (dropout off)"; go A1 "$P/jsons/${PDB}-final-updated.json" "" 0.0
echo "=== control, run 2 (dropout off) -- should match A1 exactly"; go A2 "$P/jsons/${PDB}-final-updated.json" "" 0.0
echo "=== p0 arm, 1 self-template (dropout off) -- does template count shift the draw?"; go B "$D/tmpl0/job.json" "$TENV" 0.0
echo "=== control, dropout ON x2, 3 recycles -- are the dropout decisions reproducible?"
NC=3 go C1 "$P/jsons/${PDB}-final-updated.json" "" 0.4
NC=3 go C2 "$P/jsons/${PDB}-final-updated.json" "" 0.4

echo; echo "=== VERDICT ==="
a1=$(grep -oE "sample_size=[0-9]+" "$O/A1.log" | head -1)
a2=$(grep -oE "sample_size=[0-9]+" "$O/A2.log" | head -1)
b=$(grep -oE "sample_size=[0-9]+" "$O/B.log" | head -1)
echo "  same config twice : A1 $a1 vs A2 $a2  -> $([ "$a1" = "$a2" ] && echo REPRODUCIBLE || echo DIFFERS)"
echo "  control vs p0 arm : A1 $a1 vs B  $b  -> $([ "$a1" = "$b" ] && echo SAME DRAW || echo DIFFERENT DRAW)"
c1=$(grep -oE "MC_DROPOUT decision=\w+" "$O/C1.log" | head -4 | tr '\n' ',')
c2=$(grep -oE "MC_DROPOUT decision=\w+" "$O/C2.log" | head -4 | tr '\n' ',')
echo "  dropout twice     : $([ "$c1" = "$c2" ] && echo REPRODUCIBLE || echo DIFFERS)"
echo "     C1 $c1"
echo "     C2 $c2"
