#!/usr/bin/env bash
# Re-run 9x1h's five known-good seeds with DUMP_FULL and keep the trunk state of whichever land in
# the good basin.
#
# WHY. We have a trustworthy pair-rep dump for the confident DECOY seed (300018, reproduced the sweep
# to ~0.01 ipTM) but none for a GOOD-basin run: seed 300014 re-ran into a decoy basin three separate
# times (with dump 0.018/ipTM 0.506, without dump 0.021/0.168 and 0.019/0.545). The dump was
# exonerated as the cause -- the no-dump replicates collapsed identically.
#
# THE FIVE SEEDS (9x1h CD, from the sweep; every acceptable pose on this target came from these):
#   300014 max DockQ 0.643 (3/5 acceptable)   300015 0.469 (1/5)   300038 0.377 (2/5)
#   300002 0.376 (1/5)                        300024 0.361 (1/5)
#
# THIS ALSO TESTS TRANSFER, and the hit count is the readout. Within those five seeds the sweep's
# acceptable rate was 8/25 = 32%; across all 250 poses it was 8/250 = 3.2%.
#   ~8 of 25 acceptable -> the seed number carries the basin across invocations
#   ~1 of 25            -> it does not, and the sweep's "good seeds" are just where the lottery landed
# Under the pessimistic rate the expected count is 0.8 and P(zero) is about 0.44, so a null result
# here is uninformative about the mechanism and simply means re-roll.
#
# ALL FIVE IN ONE PROCESS, matching how the sweep invoked 9x1h (50 seeds per invocation) rather than
# the one-seed-per-process form used in the failed reruns -- if RNG state carried across seeds is
# what matters, this is closer to the original conditions, and it also saves four model loads.
#
# FILE NAMING: runner/inference.py now includes DUMP_FULL in the condition that sets DUMP_REPS_TAG.
# Without that fix every seed wrote to the default tag "x" and each overwrote the last -- that bug
# already cost us one dump and was caught mid-run only by copying the file out in time.
#
# DISK: 3.3 GB per seed (1.1 GB reps + 2.2 GB feat), 16.5 GB for five, against ~120 GB of quota
# headroom. Dumps for seeds that miss the good basin are NOT deleted automatically -- they are still
# valid decoy-basin states, and deleting data on a guess is how information gets lost.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-6}
SEEDS=${SEEDS:-300014,300015,300038,300002,300024}
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/goodseeds
REPS=$OUT/reps
mkdir -p "$REPS" "$OUT/out" "$OUT/pxm_json"
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) quota before: $(quota -s 2>/dev/null | tail -1 | awk '{print $2" of "$4}')"
echo "=== $(date +%F\ %H:%M:%S) pausing pxm$GPU"
CHUNK=$(tmux capture-pane -t "pxm$GPU" -p 2>/dev/null | grep -E '^### GPU'"$GPU"' start' | tail -1 \
        | awk '{print $4"_"$6}' | sed 's/+.*//')
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 10
if [ -n "$CHUNK" ] && [ -d "$P/out/.claims_chunk/$CHUNK" ]; then
  rmdir "$P/out/.claims_chunk/$CHUNK" 2>/dev/null && echo "    released chunk claim $CHUNK"
fi

echo "=== $(date +%F\ %H:%M:%S) running seeds $SEEDS with DUMP_FULL (one process, ~20 min)"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
DUMP_FULL=1 DUMP_FULL_DIR="$REPS" \
$PY /home/ebrahim/Protenix/runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$P/jsons/9x1h-final-updated.json" \
  --dump_dir "$OUT/out" \
  --seeds "$SEEDS" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$OUT/run.log" 2>&1
echo "    exit $?  poses=$(find "$OUT/out" -name '*_sample_*.cif' 2>/dev/null | wc -l) (expect 25)"
grep -iE 'DUMP_FULL failed|Traceback' "$OUT/run.log" | head -5

echo "=== dumped reps (one pair per seed, distinct names = the tag fix worked):"
ls -lh "$REPS" 2>/dev/null | awk 'NR>1{print "    "$9"  "$5}'

echo "=== $(date +%F\ %H:%M:%S) scoring 25 poses"
for f in $(find "$OUT/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
  s=$(echo "$f" | grep -oE 'seed_[0-9]+' | grep -oE '[0-9]+')
  k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
  o=$OUT/pxm_json/${s}_${k}.json
  [ -s "$o" ] || $REPO/envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
done
echo "    scored: $(ls $OUT/pxm_json/*.json 2>/dev/null | wc -l)/25"
$PY scripts/read_goodseeds.py

echo "=== quota after: $(quota -s 2>/dev/null | tail -1 | awk '{print $2" of "$4}')"
echo "=== $(date +%F\ %H:%M:%S) handing GPU$GPU back to PXM"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
sleep 5
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "    pxm$GPU restarted" || echo "    FAILED to restart"
echo "=== $(date +%F\ %H:%M:%S) DONE"
