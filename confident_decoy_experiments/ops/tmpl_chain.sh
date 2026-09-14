#!/usr/bin/env bash
# Strict recursive self-templating: 1 seed, 1 diffusion sample, 3 iterations.
#
# WHY THIS AND NOT THE PREVIOUS RUNS. Generations 1 and 2 used 5 and 10 seeds at N_sample=5 and then
# templated on the BEST of 25 or 50 poses. That mixes two things: selection (best-of-N) and refinement
# (what templating itself buys). One seed and one sample removes selection entirely, so this is a pure
# fixed-point iteration -- x_{n+1} = model(template=x_n) -- and the trajectory is interpretable.
#
# WHAT THE OUTCOMES MEAN
#   DockQ flat, ipTM flat   -> a genuine fixed point; self-templating is stable
#   DockQ falls, ipTM rises -> the drift pathology: the donor wanders off the native basin while the
#                              model grows more certain about wherever it went. This is what
#                              generational self-templating did on ESMFold2 (0.587 -> 0.511, n=35),
#                              and it is the confident-decoy failure in another guise.
#   DockQ rises             -> refinement was not saturated after all
# Generation 2 already suggested saturation (only 3/50 poses beat their 0.945 donor, mean -0.049),
# so flat is the expectation and drift is the thing worth catching.
#
# NO SELECTION ANYWHERE: whatever the single sample produces becomes the next template, good or bad.
# That is the point -- a chain that only survives by cherry-picking is not a fixed point.
#
# A FRESH TEMPLATE ID PER ITERATION (z9x3/z9x4/z9x5). Reusing one id would leave a stale cif in
# /home/ebrahim/mmcif and silently template iteration n on iteration n-1's structure.
#
# Borrows one GPU from the PXM sweep and hands it back at the end.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-0}
P=$REPO/artifacts/pxm2025
C=$REPO/artifacts/tmplchain
START=${START:?}          # cif of the pose to start the chain from
mkdir -p "$C"
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) pausing pxm$GPU"
CHUNK=$(tmux capture-pane -t "pxm$GPU" -p 2>/dev/null | grep -E '^### GPU'"$GPU"' start' | tail -1 \
        | awk '{print $4"_"$6}' | sed 's/+.*//')
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 10
if [ -n "$CHUNK" ] && [ -d "$P/out/.claims_chunk/$CHUNK" ]; then
  rmdir "$P/out/.claims_chunk/$CHUNK" 2>/dev/null && echo "    released chunk claim $CHUNK"
fi

CUR=$START
echo "=== chain start: $CUR"
for it in 1 2 3; do
  TID="z9x$((2+it))"
  D=$C/it$it
  mkdir -p "$D/out" "$D/pxm_json"
  echo "=== $(date +%F\ %H:%M:%S) iteration $it  (template id $TID)"
  echo "    donor: $CUR"
  $PY scripts/build_bestpose_template.py --out "$D" --tid "$TID" --pose "$CUR" \
    | sed 's/^/    /' || { echo "    build FAILED"; break; }

  cd /home/ebrahim/Protenix || exit 1
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
  PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1 \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$D/job.json" \
    --dump_dir "$D/out" \
    --seeds 1 \
    --sample_diffusion.N_sample 1 --sample_diffusion.N_step 200 \
    --use_template true > "$D/run.log" 2>&1
  cd "$REPO" || exit 1
  NEW=$(find "$D/out" -name '*_sample_0.cif' | head -1)
  if [ -z "$NEW" ]; then echo "    NO POSE PRODUCED -- stopping"; break; fi
  grep -q 'Found 0 templates' "$D/run.log" && echo "    !! FOUND 0 TEMPLATES -- iteration $it ran untemplated"

  envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$NEW" -o "$D/pxm_json/score.json" >/dev/null 2>&1
  $PY - "$D" "$it" <<'EOF'
import json, os, sys, glob
D, it = sys.argv[1], sys.argv[2]
def cidx(n):
    v=0
    for c in n.split("-")[0]: v=v*26+(ord(c.upper())-64)
    return v-1
J=json.load(open(f"{D}/pxm_json/score.json"))
v=(J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
r2m=J.get("ref_to_model_chain_mapping",{})
cf=glob.glob(f"{D}/out/**/*summary_confidence_sample_0.json", recursive=True)
ip=float("nan")
if cf and all(c in r2m for c in ("C","D")):
    M=json.load(open(cf[0])).get("chain_pair_iptm")
    i,j=cidx(r2m["C"]),cidx(r2m["D"])
    if M and 0<=i<len(M) and 0<=j<len(M[i]): ip=float(M[i][j])
dq=float(v["dockq"]) if v and v.get("dockq") is not None else float("nan")
print(f"    -> iteration {it}: DockQ {dq:.3f}   ipTM {ip:.3f}")
open(f"{D}/../trajectory.csv","a").write(f"{it},{dq:.4f},{ip:.4f}\n")
EOF
  CUR=$NEW
done

echo "=== trajectory (iteration,DockQ,ipTM):"
echo "    0 (chain start): $START"
cat "$C/trajectory.csv" 2>/dev/null | sed 's/^/    /'
echo "=== $(date +%F\ %H:%M:%S) handing GPU$GPU back to PXM"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
sleep 4
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "    pxm$GPU up" || echo "    FAILED pxm$GPU"
echo "=== $(date +%F\ %H:%M:%S) DONE"
