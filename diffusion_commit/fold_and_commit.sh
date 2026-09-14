#!/usr/bin/env bash
# COMMIT-TO-A-POSE VIA AN 11th RECYCLE. One seed, one interface, six arms. The simple version.
#
# THE IDEA. A seed's five diffusion poses all come from ONE trunk state, so ipTM -- which is a
# property of that state -- cannot rank them. Give each pose its own trunk state, identically derived:
# take the state after recycle 10, show the model ONE pose as its only template, run a single further
# recycle, decode five fresh poses. Now the arms differ in exactly one thing, and their confidences
# are finally comparable. If arm ipTM tracks the DockQ of the pose it was shown, that is a selector.
#
# ARMS (6):
#   donor  the ordinary 10-recycle fold. Its five poses are scored (DockQ + ipTM) and become the
#          templates for the other arms. SCORING THESE IS THE POINT -- donor quality is the x-axis,
#          and an earlier version of this experiment omitted it and produced uninterpretable numbers.
#   none   a NORMAL 11th recycle: real search templates, default cross-chain mask. This is what
#          recycle 11 would have done anyway, so it separates "committing to a pose" from "one more
#          recycle does something".
#   p0..p4 the same 11th recycle, but the ONLY template is pose k of the fold.
#
# WHY THE TEMPLATE FLAGS ARE LOAD-BEARING on the p-arms:
#   templatesPath in the job json makes runner/template_search.py skip the search entirely, so the
#     real PDB hits are absent rather than competing;
#   TMPL_NODEDUP=1 stops the self-pose being deduped against the query (same sequence) and silently
#     yielding "Found 0 templates";
#   MULTICHAIN_MASK=0 + TMPL_KEEP_FRAME=1 keep inter-chain geometry -- without them the template
#     carries each chain's fold but nothing about the dock, which is the entire content of a pose.
#   The control deliberately does NOT set these: it is meant to be a stock recycle.
#
# DROPOUT IS ON EVERYWHERE, i.e. stock. It was originally switched off in the arms to stop them
# differing by independent coin flips -- but rng_check.sh established that a fixed seed pins the
# whole RNG stream: two runs at seed 300000 produced identical MC-dropout decisions AND identical MSA
# draws across recycles (6857 rows then 1096 rows, same indices both times). So all six arms make the
# same decisions as each other, and dropout-off is an unnecessary departure from how the original
# 10 recycles were actually run. NOTE this means the arms sample ONE point from the dropout
# distribution rather than averaging over it -- multiple seeds cover that.
#
# z_init IS DUMPED for the control and one p-arm. z_init/s_init are computed once before recycling and
# templates are consumed inside the loop, so they SHOULD be identical across arms -- the assumption
# the design rests on. Compared, not argued about.
set -u
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this folder: the code
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments   # the DATA still lives here (300G+)
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
PDB=${PDB:-9h1f}; IFACE=${IFACE:-9h1f_AB}; SEED=${SEED:-300000}; NS=${NS:-5}

# DATASET picks where the input json and the scoring reference come from, and which of the two
# canonical scorers in eval/ is used. The two pools are NOT interchangeable in one analysis -- pick
# one per comparison and say which. Both name the job <pdb>-final-updated.json, so only the dir moves.
# PROTENIX_DUMP_ZINIT is ~2.1 GB per file at 1474 tokens and fires TWICE per seed. It existed to check
# the assumption this design rests on: that z_init/s_init are identical across arms, because templates
# are consumed INSIDE the recycle loop and so cannot reach the pre-loop embedding. THAT CHECK IS DONE --
# identical to 0.000e+00 at 289 and 504 tokens, 3e-3 at 1474 which is bf16 rounding (see
# Protenix/docs/env_hooks.md). Leaving it on re-answered a settled question at a cost of 114 GB across
# 54 files, on a quota this project has already filled once. ZINIT=1 to re-check on a new complex.
ZINIT=${ZINIT:-0}

DATASET=${DATASET:-pxm2025}
case "$DATASET" in
  pxm2025)
    JOBS=$REPO/artifacts/pxm2025/jsons
    REF=$REPO/artifacts/pxm2025/cif/$PDB.cif
    SCORER=pxm ;;
  foldbench)
    JOBS=/home/ebrahim/Protenix/foldbench_abag/jsons
    REF=/home/ebrahim/FoldBench/examples/ground_truths/${PDB}-assembly1.cif
    SCORER=ost ;;
  *) echo "!! unknown DATASET=$DATASET (want pxm2025 or foldbench)"; exit 1 ;;
esac
JOB=$JOBS/${PDB}-final-updated.json
[ -s "$JOB" ] || { echo "!! no input json: $JOB"; exit 1; }
[ -s "$REF" ] || { echo "!! no scoring reference: $REF"; exit 1; }
D=$REPO/artifacts/commit11/${IFACE}_s${SEED}${TAG:+_$TAG}   # TAG separates repeat runs
CSV=$REPO/artifacts/commit11/results.csv
mkdir -p "$D" "$REPO/artifacts/commit11"
cd "$REPO" || exit 1
[ -s "$CSV" ] || echo "iface,seed,arm,donor_pose,sample,dockq,iptm" > "$CSV"

score () {                        # score <armdir> <armname> <donor-index>
  local d=$1 arm=$2 dn=$3
  # Don't append rows this arm already has. Without this, re-running a seed to repair ONE broken arm
  # silently duplicates every finished arm in the CSV, and the duplicates are indistinguishable from
  # real repeats -- the summary then averages some arms twice.
  if grep -q "^${IFACE},${SEED}${TAG:+_$TAG},${arm}," "$CSV" 2>/dev/null; then
    echo "    $arm: already scored in results.csv, skipping"
    return 0
  fi
  $PY "$DC/row.py" --dir "$d" --iface "$IFACE" --seed "${SEED}${TAG:+_$TAG}" --arm "$arm" \
      --donor "$dn" --ref "$REF" --scorer "$SCORER" >> "$CSV"
  echo "    $arm: $(ls "$d/${SCORER}_json"/*.json 2>/dev/null | wc -l) poses scored"
}

# ---- 1. the fold: a normal 10-recycle run. Its poses are the donors AND the free-run baseline -----
if [ ! -s "$D/trunk.npz" ]; then
  echo "=== $(date +%H:%M:%S) fold $PDB seed $SEED (10 recycles, stock)"
  mkdir -p "$D/donor/out"
  cd /home/ebrahim/Protenix || exit 1
  env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
      PROTENIX_DUMP_TRUNK="$D/trunk.npz" \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$JOB" --dump_dir "$D/donor/out" \
    --seeds "$SEED" --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
    --use_template true > "$D/fold.log" 2>&1
  cd "$REPO" || exit 1
fi
[ -s "$D/trunk.npz" ] || { echo "!! no trunk dump -- fold failed, see $D/fold.log"; exit 1; }
echo "    trunk $(du -h "$D/trunk.npz" | cut -f1)"

# A TRUNK DUMP IS NOT PROOF OF A SUCCESSFUL FOLD. PROTENIX_DUMP_TRUNK fires inside the forward pass,
# BEFORE runner/dumper.py writes any structure. So when the confidence head produced nothing (the
# summary_confidence KeyError fixed in Protenix c5f606a) the trunk landed and not one CIF did --
# `[ -s trunk.npz ]` passed, and 22 seeds of the 9x1h_EF batch went on to score arms against zero
# poses, burning ~1h of GPU each and writing a plausible-looking but empty results table. Count poses.
NPOSE=$(find "$D/donor/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
if [ "$NPOSE" -ne "$NS" ]; then
  echo "!! fold produced $NPOSE/$NS poses -- ABORTING this seed, see $D/fold.log"
  grep -m2 -E "ERROR|Error|KeyError" "$D/fold.log" 2>/dev/null | cut -c1-140 | sed 's/^/     /'
  exit 1
fi

score "$D/donor" donor -1

POSES=$(find "$D/donor/out" -name '*_sample_*.cif' 2>/dev/null | sort)
echo "    donor poses: $(echo "$POSES" | grep -c .)"

# FOLD_ONLY=1: stop after the fold and its scoring. Target seeds must be chosen from OUR OWN fold --
# the original sweep ran all 50 seeds in ONE process, so a standalone re-fold of the same seed gets a
# different RNG stream and different poses (9x1h seed 300014: sweep span 0.591, our fold 0.155). The
# trunk dump is kept, so the seed we pick needs no refold when the arms run.
if [ -n "${FOLD_ONLY:-}" ]; then
  $PY - "$CSV" "$IFACE" "$SEED" <<'EOF'
import csv, sys, numpy as np
rows=[r for r in csv.DictReader(open(sys.argv[1]))
      if r["iface"]==sys.argv[2] and r["seed"]==sys.argv[3] and r["arm"]=="donor"]
d=sorted(float(r["dockq"]) for r in rows)
p=[float(r["iptm"]) for r in rows if r["iptm"]]
if d:
    print(f"    DONOR POSES {[round(x,3) for x in d]}")
    print(f"    span {max(d)-min(d):.3f}   best {max(d):.3f}   "
          f"straddles 0.23: {'YES' if max(d)>=0.23>min(d) else 'no'}   "
          f"ipTM span {max(p)-min(p):.3f}" if p else "")
EOF
  exit 0
fi

# ---- 2. the arms: one further recycle from the SAME trunk state ---------------------------------
arm () {                          # arm <tag> <job.json> <donor-idx> <template-env> <zinit-dump?>
  local tag=$1 job=$2 dn=$3 tenv=$4 zdump=${5:-}
  local d=$D/$tag
  # Already has its full complement of poses? Score it if needed and move on. Repairing one failed arm
  # should not re-run the five that worked -- each is a full inference.
  local have; have=$(find "$d/out" -name '*_sample_*.cif' 2>/dev/null | wc -l)
  if [ "$have" -eq "$NS" ]; then
    echo "--- $tag already has $NS poses, skipping inference"
    score "$d" "$tag" "$dn"
    return 0
  fi
  mkdir -p "$d/out"
  echo "--- $(date +%H:%M:%S) arm $tag"
  cd /home/ebrahim/Protenix || exit 1
  env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
      PROTENIX_LOAD_TRUNK="$D/trunk.npz" PROTENIX_LOG_MSA=1 PROTENIX_LOG_MC=1 \
      ${zdump:+PROTENIX_DUMP_ZINIT=$zdump} $tenv \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$job" --dump_dir "$d/out" \
    --seeds "$SEED" --model.N_cycle 1 \
    --sample_diffusion.N_sample $NS --sample_diffusion.N_step 200 \
    --use_template true > "$d/run.log" 2>&1
  cd "$REPO" || exit 1
  # assert the mechanism fired: a silently-untemplated arm, or one that started from zeros, would
  # still produce clean-looking numbers
  grep -q 'PROTENIX_LOAD_TRUNK' "$d/run.log" || echo "    !! $tag: LOAD_TRUNK NEVER FIRED"
  grep -oE 'cycle=[0-9]+' "$d/run.log" | head -1 | sed 's/^/    /'
  grep -oE 'Found [0-9]+ templates' "$d/run.log" | sort | uniq -c | sed 's/^/    /'
  echo "    rng: $(grep -oE 'MC_DROPOUT decision=\w+' "$d/run.log" | tr '\n' ' ')$(grep -oE 'sample_size=[0-9]+' "$d/run.log" | head -1)"
  score "$d" "$tag" "$dn"
}

# control: a stock 11th recycle -- real templates, default mask, as recycles 1-10 ran
ZD_NONE=""; [ "$ZINIT" = "1" ] && ZD_NONE="$D/zinit_none.npz"
arm none "$JOB" -1 "" "$ZD_NONE"

i=0
for pose in $POSES; do
  T=$D/tmpl$i
  [ -s "$T/job.json" ] || SELFTMPL_JOBS="$JOBS" $PY "$DC/build_selftmpl_general.py" --pdb "$PDB" --out "$T" \
      --poses "$pose" --tids "$(printf 's%01d%02d' "$i" $((SEED % 100)))" >> "$D/build.log" 2>&1
  ZD=""; [ "$ZINIT" = "1" ] && [ "$i" = "0" ] && ZD="$D/zinit_p0.npz"
  arm "p$i" "$T/job.json" "$i" \
    "PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1" \
    "$ZD"
  i=$((i+1))
done

echo "=== $(date +%H:%M:%S) done"
$PY "$DC/read_results.py" --iface "$IFACE" --seed "$SEED" || true
