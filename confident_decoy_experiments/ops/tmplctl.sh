#!/usr/bin/env bash
# 9nph_AC: WHY did self-templating inflate ipTM 0.315 -> 0.824 with DockQ frozen at 0.06?
#
# TWO EXPLANATIONS FIT EVERY NUMBER WE HAVE, because Arm B changed three things at once relative to
# Arm A: it deleted the 4 real templates, added 5 self-poses, and unmasked cross-chain template
# features. The inflation may be self-corroboration (the model shown its own output and believing it)
# or simply the removal of real templates that had been holding confidence down.
#
#   run0  free      4 real templates, Arm A config, SEED 300021 -- the donor baseline (expect ~0.315)
#   run0b free      same, SEED 300046                      -- seed-matched control (expect ~0.245)
#   run1  replicate 5 self-templates, stock config        -- reproduce 0.824 / DockQ 0.061 or STOP
#   run2  addback   4 real + 5 self = 9 templates         -- falls to ~0.3 => REMOVAL was the cause
#                                                            stays ~0.82  => SELF-TEMPLATING was
#   run3  fullmsa   5 self-templates, full MSA, no dropout -- shrinks => the template was papering
#                                                            over a thin MSA; unchanged => that dies
#
# SEEDS. run1/2/3 run under 300046, the seed hybrid25 used for this templated unit. run0 runs under
# 300021 -- the DONOR seed, whose five poses ARE the templates -- because the lift under study is
# that seed's own 0.315 -> 0.824, and because its trunk state is what the templates were made from.
# run0b repeats the baseline at 300046 so the seed difference is measured rather than assumed away.
#
# RUN 1 IS A GATE. If it does not reproduce Arm B, runs 2 and 3 are uninterpretable and this script
# exits rather than burning the GPU. Output goes to artifacts/tmplctl/, never artifacts/hybrid25/,
# so the live sweep's s300021 unit is untouched.
#
# RUN 2 MASKING. The 4 real templates are pinned to their Arm A role (fold only, no dock) by
# PROTENIX_TMPL_XCHAIN_FROM=4, which applies the intra-chain mask per template_id inside
# single_template_forward. build_selftmpl_general.py --prepend-real writes them as ids 0-3 and the
# self-poses as 4-8. Without this the real templates would gain inter-chain geometry they never had,
# adding a difference we are not trying to test.
#
# DUMP_FULL saves z_trunk / s_trunk / s_inputs / coordinate / contact_probs / plddt plus the input
# feature dict for each condition, which is everything needed to re-run the CONFIDENCE HEAD offline.
# That is the point of the whole exercise: with all four dumps we can swap one condition's z_trunk
# under another condition's pose and see which input the ipTM rise actually lives in. It fires once
# per forward pass (once per seed, not per sample), so it is 5 dumps total. Each is ~0.2-0.8 GB
# depending on token count; quota headroom was 257 GB when this was written.
#
# RUN 3 FLAGS are the project's existing no-dropout recipe (ops/run_traj_nodrop.sh):
#   --data.msa.min_size.test 16384   full MSA depth instead of the randint(1,n) subsample
#   --mc_dropout_apply_rate 0.0      makes `random.random() < rate` always false
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-0}
PDB=9nph; DSEED=300021; RSEED=300046
P=$REPO/artifacts/pxm2025
S=$REPO/artifacts/tmplctl
ONLY=${ONLY:-}                      # ONLY="run0 run1" to stop after those
mkdir -p "$S"
cd "$REPO" || exit 1

POSES=$(ls "$P/out/$PDB/$PDB/seed_$DSEED/predictions/${PDB}_sample_"*.cif 2>/dev/null | sort)
[ "$(echo "$POSES" | grep -c .)" -eq 5 ] || { echo "need 5 donor poses"; exit 1; }
TIDS=$($PY -c "
import json;t=json.load(open('$REPO/artifacts/hybrid25/tids.json'))
print(' '.join(t[f'${PDB}_${DSEED}_{d}'] for d in range(5)))")
echo "donor seed $DSEED -> run seed $RSEED   tids: $TIDS"

want () { [ -z "$ONLY" ] && return 0; case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac; }

# run <tag> <job.json> <expected-ntmpl> <env> <extra-args> [seed]
run () {
  local tag=$1 job=$2 ntmpl=$3 xenv=$4 xargs=$5 seed=${6:-$RSEED}
  local D=$S/$tag
  mkdir -p "$D/out" "$D/pxm_json" "$D/full"
  echo "=== $(date +%H:%M:%S) $tag  (seed=$seed  ntmpl=$ntmpl  env='$xenv'  args='$xargs')"
  cd /home/ebrahim/Protenix || exit 1
  env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
      DUMP_FULL=1 DUMP_FULL_DIR="$D/full" $xenv \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$job" --dump_dir "$D/out" \
    --seeds "$seed" --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
    --use_template true $xargs > "$D/run.log" 2>&1
  cd "$REPO" || exit 1

  # ASSERT THE MECHANISM FIRED before trusting any number. A silently-ignored env var, or a dedup
  # collapsing 9 templates back to 4, would still produce a clean-looking ipTM.
  grep -oE 'Found [0-9]+ templates' "$D/run.log" | sort | uniq -c | sed 's/^/     /'
  grep -qE "Found $ntmpl templates" "$D/run.log" \
    || echo "     !! EXPECTED $ntmpl TEMPLATES -- $tag is not testing what it claims"
  case "$xenv" in *XCHAIN_FROM*)
    grep -q 'PROTENIX_TMPL_XCHAIN_FROM' "$D/run.log" \
      && echo "     per-template mask: $(grep -oE 'XCHAIN_FROM=[0-9]+:.*' "$D/run.log" | head -1)" \
      || echo "     !! XCHAIN_FROM SET BUT NEVER FIRED -- real templates got cross-chain geometry";;
  esac
  local nd; nd=$(ls "$D/full"/*_reps.npz 2>/dev/null | wc -l)
  [ "$nd" -ge 1 ] && echo "     DUMP_FULL: $nd reps.npz ($(du -sh "$D/full" 2>/dev/null | cut -f1))" \
                  || echo "     !! NO DUMP written for $tag"

  for f in $(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
    s=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
    [ -s "$D/pxm_json/$s.json" ] || envs/pxmeter/bin/pxm -r "$P/cif/$PDB.cif" -m "$f" \
        -o "$D/pxm_json/$s.json" >/dev/null 2>&1
  done
  $PY scripts/tmplctl_read.py --tag "$tag" | sed 's/^/     /'
}

# ---- run 0: the free baseline, on the DONOR seed (Arm A config) ---------------------------------
# No MULTICHAIN_MASK / NODEDUP / MAX_TEMPLATES overrides -- defaults are exactly what Arm A ran.
# SEED 300021, NOT 300046: its five poses ARE the templates used by run1/2/3, so this is the trunk
# state the self-templates were made from, and 0.315 -> 0.824 is the lift under study. It is also
# the dump we want for the cross-swap test (score run1's pose under run0's z_trunk).
# run0b repeats it at 300046 so the seed change between run0 and run1/2/3 stays measurable rather
# than being a silent confound -- 6 minutes of insurance against "it was just a different seed".
want run0  && run run0  "$P/jsons/${PDB}-final-updated.json" 4 "" "" "$DSEED"
want run0b && run run0b "$P/jsons/${PDB}-final-updated.json" 4 "" "" "$RSEED"

# ---- run 1: replicate Arm B exactly, and gate on it ---------------------------------------------
if want run1; then
  $PY scripts/build_selftmpl_general.py --pdb "$PDB" --out "$S/run1" --poses $POSES \
      --tids $TIDS --no-cache-write || exit 1
  run run1 "$S/run1/job.json" 5 \
      "PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=5" ""
  IPTM=$($PY scripts/tmplctl_read.py --tag run1 --field iptm 2>/dev/null)
  echo "=== gate: run1 max ipTM = ${IPTM:-none} (need >= 0.70 to match Arm B's 0.824)"
  awk -v v="${IPTM:-0}" 'BEGIN{exit !(v>=0.70)}' || {
    echo "=== GATE FAILED -- not reproducing Arm B; stopping before run2/run3"; exit 2; }
fi

# ---- run 1b: the SAME seed that produced the templates consumes them ----------------------------
# run1 feeds 300021's poses to seed 300046, so donor and consumer are different trunk states. Here
# they are the same one, which is maximally circular corroboration, and run0 is its exact matched
# baseline (same seed, same MSA draw, no templates: DockQ 0.063, ipTM 0.350). Same job.json as run1;
# only --seeds differs. ~0.82 again => the effect does not care that the template came from THIS
# seed, it just needs a confident-looking template. Higher => matched provenance matters.
want run1b && run run1b "$S/run1/job.json" 5 \
    "PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=5" \
    "" "$DSEED"

# ---- run 2: 4 real (intra-chain) + 5 self (cross-chain) -----------------------------------------
if want run2; then
  $PY scripts/real_tmpl_ids.py --job "$P/jsons/${PDB}-final-updated.json" \
      --out "$S/real_tmpl.json" || exit 1
  $PY scripts/build_selftmpl_general.py --pdb "$PDB" --out "$S/run2" --poses $POSES \
      --tids $TIDS --no-cache-write --prepend-real "$S/real_tmpl.json" || exit 1
  run run2 "$S/run2/job.json" 9 \
      "PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=9 PROTENIX_TMPL_XCHAIN_FROM=4" ""
fi

# ---- run 3: 5 self-templates, full MSA, no dropout ----------------------------------------------
if want run3; then
  mkdir -p "$S/run3"; cp "$S/run1/job.json" "$S/run3/job.json"   # identical templates to run1
  run run3 "$S/run3/job.json" 5 \
      "PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=5" \
      "--data.msa.min_size.test 16384 --mc_dropout_apply_rate 0.0"
fi

echo "=== $(date +%H:%M:%S) done"
$PY scripts/tmplctl_read.py --summary 2>/dev/null || true
