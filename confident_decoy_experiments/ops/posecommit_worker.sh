#!/usr/bin/env bash
# PER-POSE COMMITMENT, 2x2: full MSA vs query-only, 1 vs 2 templated recycles. 10 complexes x 50 seeds.
#
# THE QUESTION. Within one seed, the five diffusion poses can span DockQ 0.04-0.63 while ipTM spans
# 0.03 -- confidence cannot rank poses that share a trunk state, because ipTM is a property of that
# state. Give each pose its OWN trunk state, identically derived, and the comparison becomes fair:
# start from the same cached 10-recycle z/s, add one (or two) recycles with that pose as the only
# template, decode. If the resulting ipTMs track the donor pose's DockQ, that is a selector.
#
# PER UNIT (one complex, one seed):
#   1. fold once, stock config, N_sample=5, PROTENIX_DUMP_TRUNK -> z/s plus five donor poses. Using
#      this run's own poses rather than Arm A's keeps donor and carry-over self-consistent.
#   2. 4 control arms: an extra recycle with NO template, one per condition. Without these, any
#      change could be the extra recycle rather than the template.
#   3. 20 pose arms: 5 poses x {msa off|full} x {1|2 recycles}, each PROTENIX_LOAD_TRUNK from step 1.
#   4. score, then DELETE the dump and the CIFs.
#
# WHY DELETE. The trunk dump is ~130 MB and the CIFs ~120 MB per unit; over 500 units that is 125 GB
# against roughly 240 GB of quota headroom, and this project has already had every job on the box die
# from a full disk once. DockQ and ipTM survive in the pxm jsons, which are kilobytes.
#
# DROPOUT IS OFF IN EVERY ARM (mc_dropout_apply_rate 0) so the only thing separating arms is the
# template and the condition. The fold in step 1 is deliberately left STOCK, because it is standing in
# for the free run.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
P=$REPO/artifacts/pxm2025
M=$REPO/artifacts/posecommit
CSV=$M/results.csv
mkdir -p "$M/.claims" "$M/.done" "$M/logs"
cd "$REPO" || exit 1
[ -s "$CSV" ] || echo "pdb,iface,seed,arm,msa,ncyc,donor_pose,sample,dockq,iptm" > "$CSV"
LOG=$M/logs/gpu$GPU.log

tail -n +2 "$M/units.csv" | while IFS=, read -r pdb iface seed tokens; do
  U="${pdb}_${seed}"
  [ -e "$M/.done/$U" ] && continue
  mkdir "$M/.claims/$U" 2>/dev/null || continue
  echo "$GPU" > "$M/.claims/$U/owner"
  D=$M/$pdb/s$seed
  mkdir -p "$D"
  echo "$(date +%H:%M:%S) GPU$GPU $U ($iface, $tokens tok)" >> "$LOG"

  # ---- 1. the fold: trunk carry-over + five donor poses ------------------------------------------
  DUMP=$D/trunk.npz
  if [ ! -s "$DUMP" ]; then
    cd /home/ebrahim/Protenix || exit 1
    env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold PROTENIX_DUMP_TRUNK="$DUMP" \
    $PY runner/inference.py --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$P/jsons/${pdb}-final-updated.json" --dump_dir "$D/fold" \
      --seeds "$seed" --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
      --use_template true > "$D/fold.log" 2>&1
    cd "$REPO" || exit 1
  fi
  if [ ! -s "$DUMP" ]; then
    echo "   $U: NO TRUNK DUMP -- releasing" >> "$LOG"; rm -rf "$M/.claims/$U"; continue
  fi
  POSES=$(find "$D/fold" -name '*_sample_*.cif' 2>/dev/null | sort)
  NP=$(echo "$POSES" | grep -c .)
  echo "   dump $(du -h "$DUMP" | cut -f1), donor poses $NP" >> "$LOG"

  score () {                                 # score <dir> <arm> <msa> <ncyc> <donor>
    local d=$1 arm=$2 msa=$3 nc=$4 dn=$5
    for f in $(find "$d/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
      local k; k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
      [ -s "$d/pxm_json/$k.json" ] || envs/pxmeter/bin/pxm -r "$P/cif/$pdb.cif" -m "$f" \
          -o "$d/pxm_json/$k.json" >/dev/null 2>&1
    done
    $PY scripts/posecommit_row.py --dir "$d" --pdb "$pdb" --iface "$iface" --seed "$seed" \
        --arm "$arm" --msa "$msa" --ncyc "$nc" --donor "$dn" >> "$CSV" 2>>"$LOG"
    rm -rf "$d/out"                          # CIFs are the bulk; DockQ/ipTM already extracted
  }

  arm_run () {                               # arm_run <tag> <job.json> <msa> <ncyc> <donor> <tmplenv>
    local tag=$1 job=$2 msa=$3 nc=$4 dn=$5 tenv=$6
    local d=$D/$tag
    [ -s "$d/pxm_json/0.json" ] && return 0  # already done
    mkdir -p "$d/out" "$d/pxm_json"
    local msaenv="" msaarg=""
    if [ "$msa" = "off" ]; then msaenv="PROTENIX_MSA_NROWS=1 PROTENIX_MSA_STRATEGY=topk"
    else msaarg="--data.msa.min_size.test 16384"; fi
    cd /home/ebrahim/Protenix || exit 1
    env CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
        PROTENIX_LOAD_TRUNK="$DUMP" $msaenv $tenv \
    $PY runner/inference.py --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$job" --dump_dir "$d/out" \
      --seeds "$seed" --model.N_cycle "$nc" \
      --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
      --mc_dropout_apply_rate 0.0 $msaarg --use_template true > "$d/run.log" 2>&1
    cd "$REPO" || exit 1
    grep -q 'PROTENIX_LOAD_TRUNK' "$d/run.log" || \
      echo "   $U/$tag: !! LOAD_TRUNK NEVER FIRED (started from zeros)" >> "$LOG"
    score "$d" "$tag" "$msa" "$nc" "$dn"
  }

  for msa in off full; do
    for nc in 1 2; do
      # control: extra recycle, no template
      arm_run "none_${msa}_c${nc}" "$P/jsons/${pdb}-final-updated.json" "$msa" "$nc" "-1" ""
      i=0
      for pose in $POSES; do
        T=$D/tmpl$i
        if [ ! -s "$T/job.json" ]; then
          $PY scripts/build_selftmpl_general.py --pdb "$pdb" --out "$T" --poses "$pose" \
              --tids "$(printf 'r%01d%02d' "$i" $((seed % 100)))" >> "$D/build.log" 2>&1 || \
              { i=$((i+1)); continue; }
        fi
        arm_run "p${i}_${msa}_c${nc}" "$T/job.json" "$msa" "$nc" "$i" \
          "PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1"
        i=$((i+1))
      done
    done
  done

  n=$(grep -c ",$pdb,$iface,$seed," "$CSV" 2>/dev/null || echo 0)
  rm -f "$DUMP"; rm -rf "$D/fold/predictions" 2>/dev/null
  touch "$M/.done/$U"
  echo "$(date +%H:%M:%S) GPU$GPU $U done, rows so far $(wc -l < "$CSV")" >> "$LOG"
done
echo "=== $(date +%H:%M:%S) GPU$GPU: no units left" >> "$LOG"
