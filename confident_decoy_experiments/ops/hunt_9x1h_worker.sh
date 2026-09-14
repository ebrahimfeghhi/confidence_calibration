#!/usr/bin/env bash
# Second hunt worker: same job as ops/hunt_9x1h.sh but over a DISJOINT seed block, with a shared
# STOP sentinel so whichever worker hits first ends the other.
#
# WHY A SEPARATE FILE rather than parameterising the original: ops/hunt_9x1h.sh is executing right
# now, and bash reads a script incrementally from a byte offset -- editing a running script makes the
# live interpreter resume in shifted code. That already happened once in this project (ops/chain3.sh)
# and silently re-ran three arms.
#
# SEED BLOCKS MUST NOT OVERLAP. The original worker walks 300000..300049 in order and is at ~300007;
# this one takes 300025..300049 from the far end. Both write into artifacts/hunt/, which is safe
# because the per-seed paths are keyed by seed. When the original reaches 300025 it will find those
# seeds already generated and score them from disk rather than re-running -- no collision, and the
# work is not duplicated.
#
# THE STOP SENTINEL is checked before each seed and immediately after each scoring step. The first
# worker to hit touches it and keeps its dump; the other exits at its next check. Misses delete their
# own dumps as before, so peak disk stays ~10 GB across both workers.
#
# THIS WORKER DOES NOT RESTART A PXM SESSION on exit -- ops/handback is left to the caller, because
# the GPU it borrows (2) belongs to a different chunk claim than the original worker's (6).
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-2}
LO=${LO:-300025}
HI=${HI:-300049}
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/hunt
REPS=$OUT/reps
STOP=$OUT/STOP
mkdir -p "$REPS" "$OUT/out" "$OUT/pxm_json"
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) worker on GPU$GPU, seeds $LO..$HI"
echo "=== pausing pxm$GPU"
CHUNK=$(tmux capture-pane -t "pxm$GPU" -p 2>/dev/null | grep -E '^### GPU'"$GPU"' start' | tail -1 \
        | awk '{print $4"_"$6}' | sed 's/+.*//')
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 10
if [ -n "$CHUNK" ] && [ -d "$P/out/.claims_chunk/$CHUNK" ]; then
  rmdir "$P/out/.claims_chunk/$CHUNK" 2>/dev/null && echo "    released chunk claim $CHUNK"
fi

SEEDS=$(seq -s, "$LO" "$HI")
echo "=== $(date +%F\ %H:%M:%S) launching $(seq "$LO" "$HI" | wc -l) seeds in one process"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
DUMP_FULL=1 DUMP_FULL_DIR="$REPS" \
$PY /home/ebrahim/Protenix/runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$P/jsons/9x1h-final-updated.json" \
  --dump_dir "$OUT/out" \
  --seeds "$SEEDS" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$OUT/run_w2.log" 2>&1 &
INF=$!
echo "    inference pid $INF"

HIT=""
for seed in $(seq "$LO" "$HI"); do
  [ -e "$STOP" ] && { echo "=== $(date +%F\ %H:%M:%S) STOP set by the other worker; exiting"; break; }
  D=$OUT/out/9x1h/seed_$seed/predictions
  while :; do
    n=$(ls "$D"/9x1h_sample_*.cif 2>/dev/null | wc -l)
    [ "$n" -ge 5 ] && break
    [ -e "$STOP" ] && break
    if ! kill -0 "$INF" 2>/dev/null; then
      [ "$n" -gt 0 ] && break
      echo "=== $(date +%F\ %H:%M:%S) generator exited before seed $seed"; break 2
    fi
    sleep 20
  done
  [ -e "$STOP" ] && { echo "=== STOP set; exiting"; break; }
  sleep 3
  for f in "$D"/9x1h_sample_*.cif; do
    [ -e "$f" ] || continue
    k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
    o=$OUT/pxm_json/${seed}_${k}.json
    [ -s "$o" ] || $REPO/envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
  done
  if $PY scripts/hunt_check.py "$seed"; then
    HIT=$seed
    touch "$STOP"
    echo "=== $(date +%F\ %H:%M:%S) HIT on seed $seed (worker 2) -- STOP set, keeping dump"
    kill "$INF" 2>/dev/null; sleep 5; kill -9 "$INF" 2>/dev/null
    break
  fi
  rm -f "$REPS/9x1h_seed${seed}_reps.npz" "$REPS/9x1h_seed${seed}_feat.npz"
done
kill "$INF" 2>/dev/null; wait "$INF" 2>/dev/null

echo
[ -n "$HIT" ] && echo "=== worker 2 GOOD-BASIN DUMP: seed $HIT" || echo "=== worker 2 finished with no hit"
echo "=== $(date +%F\ %H:%M:%S) handing GPU$GPU back to PXM"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
sleep 5
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "    pxm$GPU restarted" || echo "    FAILED to restart"
echo "=== $(date +%F\ %H:%M:%S) worker 2 DONE"
