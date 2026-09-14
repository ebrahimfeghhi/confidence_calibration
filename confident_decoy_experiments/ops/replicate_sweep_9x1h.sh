#!/usr/bin/env bash
# Can the 9x1h sweep reproduce its own result? One faithful replication per GPU, stop on first hit.
#
# THE PUZZLE. The sweep (2026-09-07 22:50) found an acceptable pose in 5 of its 50 seeds; its best,
# seed 300014 sample 3, re-scores to DockQ 0.643 TODAY, so the pose is real and the scorer is stable.
# Since then 33 seed-runs across four separate re-invocations have produced zero hits (P = 0.031
# under the sweep's own 10%-per-seed rate), and every input has been audited as unchanged:
#   input json 2026-09-06 . MSA/hmmsearch a3m 2026-09-06 . template count 4 per chain in both runs
#   . none of the 27 real cifs added to ~/mmcif after the sweep appear in 9x1h's hit list
#   . post-sweep code edits are env-gated or additive . no PROTENIX_* vars set (read from /proc)
#   . 8 identical GPUs . 9x1h was claimed WHOLE-COMPLEX, i.e. 50 seeds in ONE process, as here
#
# WHAT IS LEFT. Two differences separate the sweep from every failed rerun:
#   1. DUMP_FULL=1        -- already exonerated: two no-dump replicates of seed 300014 collapsed
#                            identically to the dumped one
#   2. WORKING DIRECTORY  -- the sweep runs `cd /home/ebrahim/Protenix` then invokes
#                            `runner/inference.py` RELATIVELY; every rerun used the experiments repo
#                            as cwd with an absolute path. Constant across all 33 failures, which is
#                            exactly how a systematic cause hides.
# This script removes both: cwd and invocation are byte-identical to ops/run_pxm2025_chunked.sh:64-71
# and nothing is dumped.
#
# WHY FOUR IDENTICAL RUNS rather than four seed-blocks. A faithful replication is "50 seeds, one
# process, that cwd" -- splitting the seeds would break the very structure under test. CUDA
# nondeterminism makes the four runs independent draws, so this is 4x the sweep, not one sweep four
# times. Together they are 200 seed-runs: if the 10% rate holds, P(no hit anywhere) is 7e-10.
#
# READING THE RESULT
#   any hit  -> the sweep is reproducible and cwd is the cause; re-run with DUMP_FULL from that cwd
#   0/200    -> chance is excluded; something environmental changed after 2026-09-07 and the next
#               step is bisecting the environment, not drawing more seeds
#
# No dumps: this is a yes/no question about reachability, so it needs poses only (~275 MB per GPU),
# not 3.3 GB trunk states. Dumping comes after we know which configuration can hit.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/replicate/g$GPU
STOP=$REPO/artifacts/replicate/STOP
mkdir -p "$OUT" "$REPO/artifacts/replicate/pxm_json_g$GPU"

# EXACT sweep invocation: cwd and relative script path are the point of the experiment
cd /home/ebrahim/Protenix || exit 1

echo "=== $(date +%F\ %H:%M:%S) GPU$GPU replication starting (cwd=$(pwd))"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$P/jsons/9x1h-final-updated.json" \
  --dump_dir "$OUT" \
  --seeds "$(seq -s, 300000 300049)" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$REPO/artifacts/replicate/run_g$GPU.log" 2>&1 &
INF=$!
echo "    pid $INF"

cd "$REPO" || exit 1
for seed in $(seq 300000 300049); do
  [ -e "$STOP" ] && { echo "=== GPU$GPU: STOP set elsewhere; exiting"; break; }
  D=$OUT/9x1h/seed_$seed/predictions
  while :; do
    n=$(ls "$D"/9x1h_sample_*.cif 2>/dev/null | wc -l)
    [ "$n" -ge 5 ] && break
    [ -e "$STOP" ] && break
    kill -0 "$INF" 2>/dev/null || { [ "$n" -gt 0 ] && break; echo "=== GPU$GPU generator exited"; break 2; }
    sleep 20
  done
  [ -e "$STOP" ] && { echo "=== GPU$GPU: STOP set; exiting"; break; }
  sleep 3
  best=$($PY - "$D" "$P/cif/9x1h.cif" "$REPO/artifacts/replicate/pxm_json_g$GPU" "$seed" <<'EOF'
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
  hit=$($PY -c "print('1' if $best >= 0.23 else '0')")
  echo "    GPU$GPU seed $seed: max DockQ $best  -> $([ "$hit" = 1 ] && echo HIT || echo miss)"
  if [ "$hit" = "1" ]; then
    touch "$STOP"
    echo "=== $(date +%F\ %H:%M:%S) GPU$GPU HIT on seed $seed -- STOP set"
    break
  fi
done
kill "$INF" 2>/dev/null; sleep 3; kill -9 "$INF" 2>/dev/null; wait "$INF" 2>/dev/null
echo "=== $(date +%F\ %H:%M:%S) GPU$GPU done"
