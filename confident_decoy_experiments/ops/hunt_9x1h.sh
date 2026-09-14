#!/usr/bin/env bash
# Hunt for a GOOD-BASIN trunk state on 9x1h: run seeds 300000..300049 in sweep order, dump every
# seed, score each as it lands, delete the dump if it missed, and stop the moment one hits.
#
# WHY. We hold a trustworthy pair-rep dump for the CONFIDENT DECOY (seed 300018: rerun ipTM 0.691 vs
# sweep 0.700) but none for a good-basin run. Seed 300014 was re-run four times and never returned to
# the good basin, and re-running all five of the sweep's good seeds gave 0/25 -- which established
# that the seed NUMBER does not carry the basin across invocations (P(zero) under the selected-seed
# rate 2e-7, versus 0.27 under the global 3.2% rate). So the only way to obtain a good-basin trunk
# state is to keep drawing until one appears.
#
# ONE PROCESS, SWEEP ORDER. The original 9x1h sweep ran its 50 seeds in a single invocation; the
# failed reruns ran one or five seeds per process. If RNG state carried across seeds is what makes
# the good basin reachable, this is the configuration that has it. It also avoids 49 model loads.
#
# THE JANITOR is why this fits on disk. Dumping 50 seeds costs 3.3 GB each = 165 GB, and only ~176 GB
# is free. So a background loop watches for each seed's 5 CIFs, scores them, and either stops the run
# (hit) or deletes that seed's npz pair (miss). Peak usage stays around 10 GB because the janitor
# trails the generator by roughly one seed.
#
# EXPECTED COST. The sweep found a good pose in 5 of its 50 seeds, so ~10 seeds (~40 min) to the
# first hit, ~3.5 h worst case, and P(no hit in 50) about 0.5%.
#
# EARLY EXIT kills the inference process on the first hit. The dump for that seed is KEPT; every
# earlier seed's dump is already gone. Nothing else on the box is touched -- the other three GPUs
# stay on the PXM sweep throughout.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-6}
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/hunt
REPS=$OUT/reps
mkdir -p "$REPS" "$OUT/out" "$OUT/pxm_json"
cd "$REPO" || exit 1

SEEDS=$(seq -s, 300000 300049)
NSEED=50

echo "=== $(date +%F\ %H:%M:%S) quota before: $(quota -s 2>/dev/null | tail -1 | awk '{print $2" of "$4}')"
echo "=== $(date +%F\ %H:%M:%S) pausing pxm$GPU"
CHUNK=$(tmux capture-pane -t "pxm$GPU" -p 2>/dev/null | grep -E '^### GPU'"$GPU"' start' | tail -1 \
        | awk '{print $4"_"$6}' | sed 's/+.*//')
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 10
if [ -n "$CHUNK" ] && [ -d "$P/out/.claims_chunk/$CHUNK" ]; then
  rmdir "$P/out/.claims_chunk/$CHUNK" 2>/dev/null && echo "    released chunk claim $CHUNK"
fi

echo "=== $(date +%F\ %H:%M:%S) launching 50 seeds in ONE process"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
DUMP_FULL=1 DUMP_FULL_DIR="$REPS" \
$PY /home/ebrahim/Protenix/runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$P/jsons/9x1h-final-updated.json" \
  --dump_dir "$OUT/out" \
  --seeds "$SEEDS" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$OUT/run.log" 2>&1 &
INF=$!
echo "    inference pid $INF"

HIT=""
for seed in $(seq 300000 300049); do
  D=$OUT/out/9x1h/seed_$seed/predictions
  # wait for this seed's 5 poses, or for the generator to die
  while :; do
    n=$(ls "$D"/9x1h_sample_*.cif 2>/dev/null | wc -l)
    [ "$n" -ge 5 ] && break
    if ! kill -0 "$INF" 2>/dev/null; then
      # generator gone; take whatever this seed has, then stop
      [ "$n" -gt 0 ] && break
      echo "=== $(date +%F\ %H:%M:%S) generator exited before seed $seed"
      break 2
    fi
    sleep 20
  done
  sleep 3                                  # let the last file finish flushing
  for f in "$D"/9x1h_sample_*.cif; do
    [ -e "$f" ] || continue
    k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
    o=$OUT/pxm_json/${seed}_${k}.json
    [ -s "$o" ] || $REPO/envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
  done
  if $PY scripts/hunt_check.py "$seed"; then
    HIT=$seed
    echo "=== $(date +%F\ %H:%M:%S) HIT on seed $seed -- stopping the generator"
    kill "$INF" 2>/dev/null; sleep 5; kill -9 "$INF" 2>/dev/null
    break
  fi
  # miss: reclaim the 3.3 GB, keep the scores and poses (they are small and tell us the hit rate)
  rm -f "$REPS/9x1h_seed${seed}_reps.npz" "$REPS/9x1h_seed${seed}_feat.npz"
  echo "      dropped dump for $seed  (free: $(quota -s 2>/dev/null | tail -1 | awk '{print $2}'))"
done
wait "$INF" 2>/dev/null

echo
if [ -n "$HIT" ]; then
  echo "=== GOOD-BASIN DUMP: seed $HIT"
  ls -lh "$REPS" | awk 'NR>1{print "    "$9"  "$5}'
else
  echo "=== NO HIT in the seeds that ran; all dumps dropped"
fi
echo "=== seeds scored: $(ls $OUT/pxm_json/*.json 2>/dev/null | sed 's/.*\///;s/_.*//' | sort -u | wc -l)"
echo "=== quota after: $(quota -s 2>/dev/null | tail -1 | awk '{print $2" of "$4}')"

echo "=== $(date +%F\ %H:%M:%S) handing GPU$GPU back to PXM"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
sleep 5
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "    pxm$GPU restarted" || echo "    FAILED to restart"
echo "=== $(date +%F\ %H:%M:%S) DONE"
