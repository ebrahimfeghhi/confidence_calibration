#!/usr/bin/env bash
# Why did 9x1h seed 300014 lose its good pose on rerun?
#
# THE OBSERVATION. The sweep's seed 300014 produced DockQ 0.643 (CD) with pair_iptm 0.131 and wide
# sample-to-sample spread. Re-running that seed under DUMP_FULL gave 0.018 on all five samples, zero
# spread, pair_iptm 0.506 -- it behaves like a decoy seed. Seed 300018 re-ran essentially exactly
# (0.019 -> 0.019, iptm 0.700 -> 0.691), so the pipeline is not broadly irreproducible.
#
# THE CODE SAYS THE DUMP IS INNOCENT: the DUMP_FULL block sits AFTER sample_diffusion and AFTER
# run_confidence_head, and 300014 was the first seed in the process, so no dump code had executed
# when its poses were made. This probe checks that empirically instead of trusting the reading.
#
# DESIGN: seed 300014 alone, NO DUMP_FULL, twice, in two SEPARATE processes. Everything else matches
# the sweep's invocation (protenix-v2 defaults: N_cycle=10, N_step=200, N_sample=5, templates on).
#   both ~0.018  -> dump exonerated AND the collapse is reproducible => seed POSITION in the process
#                   is the difference (the sweep ran 9x1h's 50 seeds in one invocation, so 300014 was
#                   the 15th seed there and the 1st here), i.e. RNG state carried across seeds
#   they differ  -> basin-commitment lottery under CUDA nondeterminism (already documented on 7txw),
#                   so re-roll until it lands in the good basin, then dump
#   either ~0.64 -> the dump IS the cause, and the code reading above is wrong
#
# No dumping here: this is a 2 x 5-pose question, and the npz files are 3.3 GB per seed.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-6}
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/probe14
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) pausing pxm$GPU"
CHUNK=$(tmux capture-pane -t "pxm$GPU" -p 2>/dev/null | grep -E '^### GPU'"$GPU"' start' | tail -1 \
        | awk '{print $4"_"$6}' | sed 's/+.*//')
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 10
if [ -n "$CHUNK" ] && [ -d "$P/out/.claims_chunk/$CHUNK" ]; then
  rmdir "$P/out/.claims_chunk/$CHUNK" 2>/dev/null && echo "    released chunk claim $CHUNK"
fi

for rep in 1 2; do
  echo "=== $(date +%F\ %H:%M:%S) replicate $rep: seed 300014 alone, no dump"
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  $PY /home/ebrahim/Protenix/runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$P/jsons/9x1h-final-updated.json" \
    --dump_dir "$OUT/rep$rep" \
    --seeds 300014 \
    --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
    --use_template true > "$OUT/rep$rep.log" 2>&1
  echo "    exit $? poses=$(find "$OUT/rep$rep" -name '*_sample_*.cif' 2>/dev/null | wc -l)"
done

echo "=== $(date +%F\ %H:%M:%S) scoring both replicates"
mkdir -p "$OUT/pxm_json"
for rep in 1 2; do
  for f in $(find "$OUT/rep$rep" -name '*_sample_*.cif' 2>/dev/null | sort); do
    k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
    o=$OUT/pxm_json/rep${rep}_${k}.json
    [ -s "$o" ] || $REPO/envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
  done
done
$PY scripts/read_probe14.py

echo "=== $(date +%F\ %H:%M:%S) handing GPU$GPU back to PXM"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
sleep 5
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "    pxm$GPU restarted" || echo "    FAILED to restart"
echo "=== $(date +%F\ %H:%M:%S) DONE"
