#!/usr/bin/env bash
# Re-run selected 9x1h seeds with DUMP_REPS so we get BOTH confidence scores per pose:
#   cpi_norm  chain_pair_iptm as normally computed
#   cpi_abl   the same head run with use_embedding=False, i.e. the trunk PAIR REP zeroed
#
# WHY. ipTM is a seed-level quantity: 87% of its variance across 262 PXM interfaces is BETWEEN seeds,
# and on 9x1h it is ~100% -- the five samples of a seed get the same score to three decimals while
# their DockQ spans 0.04-0.64. We traced the seed-level signal to MSA subsampling inside the model
# (sample_indices in protenix/model/utils.py draws a random SIZE and a random SUBSET per seed), which
# reaches the head only through z_trunk. Zeroing z_trunk should therefore strip the seed signal and
# leave the head reading the only per-pose input it has: the binned distance map. That map is NOT the
# bottleneck -- it tracks structural difference at r = 0.873 vs DockQ.
#
# PREDICTION: cpi_abl should vary WITHIN a seed where cpi_norm does not, and should rank 9x1h's good
# poses above its decoys. If it does, that is a better selector for free -- one extra forward pass
# through a 4-block head, no retraining, no extra sampling.
#
# use_embedding=False is a TRAINED condition, not an off-distribution hack: training drops the
# embedding at rate confidence_embedding_drop_rate (protenix.py:1209), so the head has learned to work
# without it.
#
# SEEDS: the 5 that produce an acceptable pose (300014/300015/300038/300002/300024, every one of them
# scattered and sitting at ipTM ~0.13 against a 0.39 mean) plus 300018, the MOST confident seed
# (ipTM 0.700, all five samples locked on the decoy at DockQ 0.02). That contrast is the whole test.
#
# ONE PROCESS PER SEED: DUMP_REPS_TAG is read once per forward pass, so several seeds in one process
# would overwrite each other's npz.
#
# CONFIG MATCHES ops/run_pxm2025.sh EXACTLY (no --model.N_cycle override -- the sweep used the config
# default) so these poses are comparable to the ones already scored. Note CUDA nondeterminism means
# the poses will not be bit-identical to the originals; the comparison that matters is cpi_norm vs
# cpi_abl WITHIN this run, which is self-contained.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/reps_9x1h
GPU=${GPU:?set GPU}
SEEDS=${SEEDS:?comma list}
mkdir -p "$OUT/out"

cd /home/ebrahim/Protenix || exit 1
for s in ${SEEDS//,/ }; do
  echo "### 9x1h seed $s on gpu$GPU $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  DUMP_REPS=1 DUMP_REPS_DIR="$OUT" DUMP_REPS_TAG="9x1h_$s" \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$P/jsons/9x1h-final-updated.json" \
    --dump_dir "$OUT/out/$s" \
    --seeds "$s" \
    --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
    --use_template true >> "$OUT/9x1h_$s.log" 2>&1
  echo "### seed $s done rc=$? npz=$(ls -1 $OUT/9x1h_$s.npz 2>/dev/null | wc -l) $(date +%H:%M:%S)"
done
echo "### GPU$GPU ALL SEEDS DONE"
