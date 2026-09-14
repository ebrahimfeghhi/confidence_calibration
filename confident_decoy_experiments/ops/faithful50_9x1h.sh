#!/usr/bin/env bash
# 9x1h with template 1nzi INCLUDED (faithful): 50 seeds split across 4 GPUs (~13 each).
#
# WHY 1nzi. Protenix keeps the first 4 templates it can load, in hit order
# (_reduce_template_features: keep_mask = arange(num_t) < max_templates -- no scoring, no re-ranking).
# 1nzi is rank 0 of the 29 loadable hits for chain 1, the antibody HEAVY chain, so it occupies the
# first template slot and pushes 3dem out of the top 4:
#     with 1nzi : 1nzi, 4aqb, 5ckq, 1nt0
#     without   : 4aqb, 5ckq, 1nt0, 3dem
# The heavy chain is one side of the C-D interface being measured, so this is a real change to the
# conditioning signal on the half of the complex we care about.
#
# HONEST CAVEAT: the sweep DID use 1nzi. Its log records "Fetching mmCIF for 1nzi" at 22:05:25, three
# seconds before its first seed, and _fetch_or_read_cif downloads and returns the content in the same
# call -- there is no download-then-skip race. So this arm tests a configuration the sweep never ran.
# It is a perturbation experiment ("does dropping the top heavy-chain template free the good basin?"),
# NOT an explanation of why the sweep's result will not reproduce.
#
# DELETING THE FILE WOULD NOT WORK. fetch_remote is enabled -- that is how 1nzi arrived in the first
# place -- so removing /home/ebrahim/mmcif/1nzi.cif just triggers a re-download from PDBe and the
# ablation silently does nothing. PROTENIX_TMPL_EXCLUDE (added to template_utils._fetch_or_read_cif,
# env-gated, raises the FileNotFoundError the caller already handles) is what actually drops it.
#
# SEEDS ARE SPLIT, NOT REPLICATED: 50 seeds once, ~13 per GPU, because the question here is the hit
# rate over the seed range rather than a faithful replication of the sweep's single-process structure.
#
# The run ASSERTS the exclusion fired -- a silent no-op ablation is the failure mode this project has
# hit repeatedly ("Found 1 templates", the a3m dedup, the DUMP_REPS_TAG collision).
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
LO=${LO:?}
HI=${HI:?}
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/faithful50/g$GPU
JD=$REPO/artifacts/faithful50/pxm_json
mkdir -p "$OUT" "$JD"
cd /home/ebrahim/Protenix || exit 1

echo "=== $(date +%F\ %H:%M:%S) GPU$GPU seeds $LO..$HI, 1nzi INCLUDED (faithful) (cwd=$(pwd))"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$P/jsons/9x1h-final-updated.json" \
  --dump_dir "$OUT" \
  --seeds "$(seq -s, "$LO" "$HI")" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$REPO/artifacts/faithful50/run_g$GPU.log" 2>&1 &
INF=$!

# assert NO exclusion is in play -- this arm must be the sweep's exact template set
sleep 150
if grep -q 'TMPL_EXCLUDE' "$REPO/artifacts/faithful50/run_g$GPU.log" 2>/dev/null; then
  echo "    !! a template was EXCLUDED -- this arm is not faithful; killing"
  kill "$INF" 2>/dev/null; exit 1
fi
echo "    OK no exclusions; Found-4 lines: $(grep -c 'Found 4 templates' "$REPO/artifacts/faithful50/run_g$GPU.log")"

cd "$REPO" || exit 1
for seed in $(seq "$LO" "$HI"); do
  D=$OUT/9x1h/seed_$seed/predictions
  while :; do
    n=$(ls "$D"/9x1h_sample_*.cif 2>/dev/null | wc -l)
    [ "$n" -ge 5 ] && break
    kill -0 "$INF" 2>/dev/null || { [ "$n" -gt 0 ] && break; echo "=== GPU$GPU generator exited"; break 2; }
    sleep 20
  done
  sleep 3
  best=$($PY - "$D" "$P/cif/9x1h.cif" "$JD" "$seed" <<'EOF'
import glob, json, os, subprocess, sys
D, REF, JD, seed = sys.argv[1:5]
best = -1.0
for f in sorted(glob.glob(f"{D}/9x1h_sample_*.cif")):
    k = os.path.basename(f).split("sample_")[1].split(".")[0]
    o = f"{JD}/{seed}_{k}.json"
    if not os.path.exists(o) or os.path.getsize(o) == 0:
        subprocess.run(["/home/ebrahim/tts_abag/confident_decoy_experiments/envs/pxmeter/bin/pxm",
                        "-r", REF, "-m", f, "-o", o],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        J = json.load(open(o))
    except Exception:
        continue
    v = (J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
    if v and v.get("dockq") is not None:
        best = max(best, float(v["dockq"]))
print(f"{best:.4f}")
EOF
)
  hit=$($PY -c "print('HIT' if $best >= 0.23 else 'miss')")
  echo "    GPU$GPU seed $seed: max DockQ $best -> $hit"
done
wait "$INF" 2>/dev/null
echo "=== $(date +%F\ %H:%M:%S) GPU$GPU done"
