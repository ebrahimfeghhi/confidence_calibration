#!/usr/bin/env bash
# Recursive self-templating, generation 2: template on generation 1's BEST pose and run 10 seeds.
#
# GENERATION 1 templated on the best FREE pose (seed 300014 sample 0, DockQ 0.683) and returned
# 25/25 acceptable, mean DockQ 0.788, max 0.945, ipTM 0.537 (0.429-0.697) -- against the free run's
# 0.120 for a good seed and 0.700 for the confident decoy. So conditioning the trunk on the answer
# both locks the pose in and makes the head confident about it.
#
# THE QUESTION HERE: does feeding the improved pose back in keep improving, or has it converged?
#   still rising  -> self-templating is a usable iterative refinement on this target
#   flat          -> generation 1 already saturated what the template can buy
#   falling       -> the donor pose is drifting off the native basin while confidence stays high,
#                    which is the failure mode generational self-templating showed on ESMFold2
#                    (selected DockQ 0.587 -> 0.511 across generations, n=35)
# The last case is why this is worth running rather than assuming monotone improvement.
#
# DONOR: 9x1h_tmplbest seed 4 sample 1, DockQ 0.945 -- the best pose generation 1 produced.
#
# SAME FLAGS AS GENERATION 1, all four load-bearing:
#   PROTENIX_MULTICHAIN_MASK=0   else cross-chain template features are zeroed and the dock discarded
#   PROTENIX_TMPL_KEEP_FRAME=1   else each chain is centred on its own centroid, collapsing the template
#   PROTENIX_TMPL_NODEDUP=1 + PROTENIX_MAX_TEMPLATES=1   else the self-pose dedups against the query
# A distinct template id (z9x2) so it cannot overwrite generation 1's entry in ~/mmcif.
#
# 10 seeds x 5 samples = 50 poses, against generation 1's 5 x 5 = 25. No dumps: this is a DockQ/ipTM
# question, and 3.3 GB per seed is not worth spending before we know the trend.
#
# Borrows GPU0 from the PXM sweep and hands it back at the end.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-0}
P=$REPO/artifacts/pxm2025
G=$REPO/artifacts/tmplgen2
DONOR=$REPO/artifacts/tmplbest/out/9x1h_tmplbest/seed_4/predictions/9x1h_tmplbest_sample_1.cif
mkdir -p "$G/out" "$G/pxm_json"
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) pausing pxm$GPU"
CHUNK=$(tmux capture-pane -t "pxm$GPU" -p 2>/dev/null | grep -E '^### GPU'"$GPU"' start' | tail -1 \
        | awk '{print $4"_"$6}' | sed 's/+.*//')
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 10
if [ -n "$CHUNK" ] && [ -d "$P/out/.claims_chunk/$CHUNK" ]; then
  rmdir "$P/out/.claims_chunk/$CHUNK" 2>/dev/null && echo "    released chunk claim $CHUNK"
fi

echo "=== $(date +%F\ %H:%M:%S) building the gen-2 template from gen-1's best pose"
$PY scripts/build_bestpose_template.py --out "$G" --tid z9x2 --pose "$DONOR" \
  || { echo "    template build FAILED"; exit 1; }

echo "=== $(date +%F\ %H:%M:%S) 10 seeds, templated on the 0.945 pose"
cd /home/ebrahim/Protenix || exit 1
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1 \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$G/job.json" \
  --dump_dir "$G/out" \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$G/run.log" 2>&1
echo "    exit $?  poses=$(find "$G/out" -name '*_sample_*.cif' 2>/dev/null | wc -l) (expect 50)"
grep -m3 -oE 'Found [0-9]+ templates' "$G/run.log" | sed 's/^/    /'
grep -q 'Found 0 templates' "$G/run.log" && echo "    !! FOUND 0 TEMPLATES -- ran UNTEMPLATED, results void"

cd "$REPO" || exit 1
echo "=== $(date +%F\ %H:%M:%S) scoring"
for f in $(find "$G/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
  s=$(echo "$f" | grep -oE 'seed_[0-9]+' | grep -oE '[0-9]+')
  k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
  o=$G/pxm_json/${s}_${k}.json
  [ -s "$o" ] || envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
done
$PY scripts/read_gen.py

echo "=== $(date +%F\ %H:%M:%S) handing GPU$GPU back to PXM"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
sleep 4
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "    pxm$GPU up" || echo "    FAILED pxm$GPU"
echo "=== $(date +%F\ %H:%M:%S) DONE"
