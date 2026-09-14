#!/usr/bin/env bash
# Dump the full trunk state of 9x1h's BEST seed and its MOST CONFIDENT seed, then prove the dump
# did not disturb the run.
#
# THE TWO SEEDS (identical picks on all four 9x1h interfaces, so there is no chain-pair ambiguity):
#   300014  best      max DockQ 0.643 (CD, sample 3)   mean pair_iptm 0.131
#   300018  confident mean pair_iptm 0.700             max DockQ 0.019
# That pair IS the confident-decoy phenomenon on this target: the seed that finds the answer is the
# one the model disbelieves.
#
# WHY NO CONFIG FLAGS. protenix-v2 already sets N_cycle=10 and N_step=200, and the PXM sweep never
# overrode them, so this reproduces the sweep's configuration exactly by NOT passing anything extra.
# The invocation below is copied from ops/run_pxm2025_chunked.sh line-for-line apart from the seeds,
# the dump dir, and DUMP_FULL.
#
# WHAT DUMP_FULL WRITES (protenix/model/protenix.py, env-gated, verified additive: 42 lines added,
# 0 removed, and the default path never enters the block):
#   <tag>_reps.npz  z_trunk [N_token,N_token,256] fp16 ~1.1 GB, s_trunk, s_inputs, the 5 sampled
#                   coordinate sets, contact_probs, plddt
#   <tag>_feat.npz  the input_feature_dict as the heads see it (msa/template features are already
#                   deleted upstream), i.e. the token/atom layout needed to re-run either head
# It fires once per forward pass and one forward pass emits all 5 samples, so z is written once per
# seed, not once per sample.
#
# THE VALIDATION. CUDA nondeterminism means this will NOT reproduce the original poses bit-for-bit
# (established earlier in this project: the 7txw input mask is reproducible per seed, the end-to-end
# output is not). So the check is distributional, not exact -- scripts/compare_dump_vs_original.py
# puts the rerun's DockQ and pair_iptm next to the sweep's for the same two seeds. What would signal
# a real problem is a shift far beyond the sweep's own sample-to-sample spread, e.g. the confident
# seed no longer scoring ~0.70, or the best seed losing its 0.6 pose entirely.
#
# GPU: pauses ONE pxm worker (default GPU6, chosen because it had just started a chunk), releases
# that chunk claim so the work is not stranded, and restarts the worker at the end. The chunked
# runner is resumable per seed, so nothing generated so far is lost.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-6}
SEEDS=${SEEDS:-300014,300018}
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/dump9x1h
REPS=$OUT/reps
mkdir -p "$REPS" "$OUT/out"

cd "$REPO" || exit 1
echo "=== $(date +%F\ %H:%M:%S) pausing pxm$GPU"
CHUNK=$(tmux capture-pane -t "pxm$GPU" -p 2>/dev/null | grep -E '^### GPU'"$GPU"' start' | tail -1 \
        | awk '{print $4"_"$6}' | sed 's/+.*//')   # "### GPU6 start 9quw seeds 300020+10 (..." -> 9quw_300020
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 10
if [ -n "$CHUNK" ] && [ -d "$P/out/.claims_chunk/$CHUNK" ]; then
  rmdir "$P/out/.claims_chunk/$CHUNK" 2>/dev/null \
    && echo "    released chunk claim $CHUNK (resumes per seed, nothing lost)"
fi
echo "    free VRAM on GPU$GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $GPU)"

echo "=== $(date +%F\ %H:%M:%S) running seeds $SEEDS with DUMP_FULL"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
DUMP_FULL=1 DUMP_FULL_DIR="$REPS" \
$PY /home/ebrahim/Protenix/runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$P/jsons/9x1h-final-updated.json" \
  --dump_dir "$OUT/out" \
  --seeds "$SEEDS" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$OUT/run.log" 2>&1
rc=$?
echo "=== inference exit $rc"
grep -iE 'DUMP_FULL failed|Traceback' "$OUT/run.log" | head -5

echo "=== dumped files:"
ls -lh "$REPS" 2>/dev/null | awk '{print "    "$9"  "$5}'
echo "=== poses produced: $(find "$OUT/out" -name '*_sample_*.cif' | wc -l) (expect 10)"

echo "=== $(date +%F\ %H:%M:%S) scoring the rerun against the reference"
mkdir -p "$OUT/pxm_json"
for f in $(find "$OUT/out" -name '*_sample_*.cif' | sort); do
  s=$(echo "$f" | grep -oE 'seed_[0-9]+' | grep -oE '[0-9]+')
  k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
  o=$OUT/pxm_json/${s}_${k}.json
  [ -s "$o" ] || $REPO/envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
done
echo "    scored: $(ls $OUT/pxm_json/*.json 2>/dev/null | wc -l)"
$PY scripts/compare_dump_vs_original.py

echo "=== $(date +%F\ %H:%M:%S) handing GPU$GPU back to PXM"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
sleep 5
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "    pxm$GPU restarted" || echo "    FAILED to restart pxm$GPU"
echo "=== $(date +%F\ %H:%M:%S) DONE"
