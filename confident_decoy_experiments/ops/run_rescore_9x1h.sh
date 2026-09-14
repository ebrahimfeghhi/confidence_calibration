#!/usr/bin/env bash
# Overnight: rescore all 250 9x1h poses by self-templating each one, at N_cycle=1 then N_cycle=3,
# then give every GPU back to the PXM-2025 sweep.
#
# THE QUESTION. ipTM as normally produced is ~100% a SEED property on 9x1h -- the five samples of a
# seed score alike however different their structures are -- because the confidence head is dominated
# by z_trunk, which carries that seed's MSA subsample. Rescoring hands each pose BACK as a template
# with its full inter-chain geometry and reads the ipTM that results, so every pose is judged from its
# own geometry rather than from whatever MSA draw produced it. Entirely label-free: if the acceptable
# poses rank top, it is a usable selector.
#
# WHY SHORT RECYCLING. At N_cycle=10 the model drifts off the template back to what it prefers
# (templating the decoy gave 0.724, a good pose 0.460) -- that is preference, not evaluation. At
# N_cycle=1 it can barely move, so the score should reflect the input. N=3 brackets the drift.
#
# TEMPLATES WERE VALIDATED BEFORE LAUNCH: per-pose protomer chains measured from contacts (the docked
# triple varies -- A/C/F 79, B/D/E 70, A/C/E 52, B/D/F 49 of 250, so it cannot be assumed), and 8
# randomly sampled entries loaded 3/3 with real inter-chain geometry (light-heavy ~17 A, antigen
# 40-67 A). With per-chain centering left on, those separations would all be 0.
#
# FIVE SEEDS PER POSE, the SAME five for every pose. One draw per pose is noisy -- the rescored ipTM
# is itself a stochastic forward pass -- and using an identical seed set for every pose makes the
# comparison paired, so seed variation is a common factor rather than a per-pose confound. Five is
# what fits: at 12.8 s/pose/seed across 4 GPUs, 5 seeds is ~10 h for both passes, 10 seeds ~20 h.
# N_sample=1 because only a confidence readout is needed, not new poses. One JSON per shard so the
# checkpoint loads 4 times, not 250 times.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
RS=$REPO/artifacts/rescore/rs
GPUS="0 1 2 6"

cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) stopping PXM workers and releasing in-flight claims"
for g in $GPUS; do tmux kill-session -t "pxm$g" 2>/dev/null; done
sleep 12
for p in $(ls artifacts/pxm2025/out/.claims 2>/dev/null); do
  [ "$p" = "9r10" ] && continue
  c=$(find "artifacts/pxm2025/out/$p" -name '*_sample_*.cif' 2>/dev/null | wc -l)
  if [ "$c" -lt 250 ]; then rmdir "artifacts/pxm2025/out/.claims/$p" 2>/dev/null \
      && echo "    released $p ($c/250 kept, resumes per seed)"; fi
done
# chunked runner claims per seed-chunk; clear any partial chunk claims too
for d in artifacts/pxm2025/out/.claims_chunk/*; do [ -d "$d" ] && rmdir "$d" 2>/dev/null; done

run_pass () {                       # $1 = N_cycle, $2 = tag
  local NC=$1 TAG=$2 i=0
  echo "=== $(date +%F\ %H:%M:%S) pass $TAG (N_cycle=$NC)"
  for g in $GPUS; do
    local SH=$RS/shard$i.json
    [ -s "$SH" ] || { i=$((i+1)); continue; }
    tmux new-session -d -s "rs_${TAG}_$g" -c /home/ebrahim/Protenix \
      "CUDA_VISIBLE_DEVICES=$g LAYERNORM_TYPE=openfold \
       PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
       PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1 \
       $PY runner/inference.py --model_name protenix-v2 \
         --load_checkpoint_dir /home/ebrahim/checkpoint \
         --input_json_path $SH \
         --dump_dir $REPO/artifacts/rescore_out/$TAG \
         --seeds 1,2,3,4,5 --model.N_cycle $NC \
         --sample_diffusion.N_sample 1 --sample_diffusion.N_step 200 \
         --use_template true >> $REPO/artifacts/rescore_$TAG.log 2>&1"
    i=$((i+1))
  done
  sleep 60
  while :; do
    local alive=0
    for g in $GPUS; do tmux has-session -t "rs_${TAG}_$g" 2>/dev/null && alive=$((alive+1)); done
    [ "$alive" -eq 0 ] && break
    sleep 60
  done
  echo "=== $(date +%F\ %H:%M:%S) pass $TAG done: \
$(find "$REPO/artifacts/rescore_out/$TAG" -name '*_summary_confidence_*.json' 2>/dev/null | wc -l) confidence files"
  grep -ohiE "Found [0-9]+ templates" "$REPO/artifacts/rescore_$TAG.log" 2>/dev/null \
    | sort | uniq -c | sed 's/^/    /'
}

run_pass 1 n1
run_pass 3 n3

echo "=== $(date +%F\ %H:%M:%S) handing all GPUs back to PXM (chunked runner)"
for g in $GPUS; do
  tmux new-session -d -s "pxm$g" -c "$REPO" "GPU=$g bash ops/run_pxm2025_chunked.sh"
done
sleep 5
echo "pxm workers up: $(tmux ls -F '#{session_name}' 2>/dev/null | grep -c '^pxm[0-9]')"
echo "=== $(date +%F\ %H:%M:%S) ALL DONE"
