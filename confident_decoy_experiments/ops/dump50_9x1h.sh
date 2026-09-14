#!/usr/bin/env bash
# 9x1h: 50 seeds total split across 4 GPUs, DUMP_FULL on, keep the dump only when the seed HITS.
#
# THE GOAL. We hold a trustworthy trunk dump for the CONFIDENT DECOY (seed 300018) and none for a
# good-basin seed. Today's faithful arm finally produced a hit -- seed 300014, DockQ 0.683, beating
# the sweep's own 0.643 -- but that arm ran without dumping, so the trunk state is gone. This repeats
# it with DUMP_FULL so a hit actually yields z_trunk/s_trunk/s_inputs.
#
# THE CONFIGURATION IS THE ONE THAT WORKS: cwd=/home/ebrahim/Protenix with a relative
# runner/inference.py path. Runs launched from the experiments repo went 0-for-66 today; the two
# runs that reached the good basin (the 2026-09-07 sweep and this afternoon's faithful arm) both used
# this cwd. Why it matters is still unexplained -- PROTENIX_ROOT_DIR resolves to $HOME either way, so
# something else is cwd-sensitive. Treated as an empirical requirement, not an understood one.
#
# DUMP_FULL IS SAFE: two no-dump replicates of seed 300014 collapsed identically to the dumped one,
# and the dump block runs after sample_diffusion and after run_confidence_head, so it cannot affect
# the poses of its own forward pass.
#
# THE JANITOR is what makes this fit: 50 dumps at 3.3 GB each is 165 GB. Each seed is scored as soon
# as its 5 poses land; a miss has its npz pair deleted immediately, a hit is kept. Peak stays near
# 10 GB. Seed blocks are disjoint across GPUs so the per-seed dump filenames cannot collide.
#
# NO EARLY STOP -- all 50 seeds run. One hit gives the dump; the full 50 also gives a hit RATE to set
# against the sweep's 5/50, which is the number that says whether this configuration really matches.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
LO=${LO:?}
HI=${HI:?}
P=$REPO/artifacts/pxm2025
OUT=$REPO/artifacts/dump50/g$GPU
REPS=$REPO/artifacts/dump50/reps
JD=$REPO/artifacts/dump50/pxm_json
mkdir -p "$OUT" "$REPS" "$JD"
cd /home/ebrahim/Protenix || exit 1

echo "=== $(date +%F\ %H:%M:%S) GPU$GPU seeds $LO..$HI, DUMP_FULL on (cwd=$(pwd))"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
DUMP_FULL=1 DUMP_FULL_DIR="$REPS" \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$P/jsons/9x1h-final-updated.json" \
  --dump_dir "$OUT" \
  --seeds "$(seq -s, "$LO" "$HI")" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$REPO/artifacts/dump50/run_g$GPU.log" 2>&1 &
INF=$!

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
  if [ "$($PY -c "print(1 if $best >= 0.23 else 0)")" = "1" ]; then
    echo "    GPU$GPU seed $seed: max DockQ $best -> HIT  (dump KEPT)"
    ls -lh "$REPS/9x1h_seed${seed}_reps.npz" 2>/dev/null | awk '{print "        "$9"  "$5}'
  else
    rm -f "$REPS/9x1h_seed${seed}_reps.npz" "$REPS/9x1h_seed${seed}_feat.npz"
    echo "    GPU$GPU seed $seed: max DockQ $best -> miss (dump dropped)"
  fi
done
wait "$INF" 2>/dev/null
echo "=== $(date +%F\ %H:%M:%S) GPU$GPU done"
