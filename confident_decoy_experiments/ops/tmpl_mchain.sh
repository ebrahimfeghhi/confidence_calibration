#!/usr/bin/env bash
# Recursive self-templating on the WHOLE round: all 5 poses of a round become the templates for the next.
#
# vs THE 1-POSE CHAIN (ipTM 0.107 -> 0.425 -> 0.452 -> 0.513, DockQ 0.683 -> 0.588 -> 0.660 -> 0.706):
# that chain fed back a single sample. This feeds back all 5, so the model sees the round's whole
# distribution -- which for a real free seed is MIXED (300014's five samples were 0.683, 0.562, 0.031,
# 0.033, 0.023). The question is whether the ensemble converges on the good poses or is dragged toward
# the bad ones. Still zero selection: every pose goes back, good or bad.
#
# WHY THIS IS NOT THE SAME AS GEN 1/2: those templated on the single BEST of 25 or 50 poses, so their
# gains mix selection with templating. Here nothing is chosen.
#
# FLAGS. PROTENIX_MAX_TEMPLATES=5 because the trained default is 4 and would silently drop a donor.
# PROTENIX_TMPL_NODEDUP=1 because the feature-level dedup keys on the mmcif sequence and would collapse
# five conformations of the same molecule into one ("Found 1 templates"). The builder separately
# guards the a3m-level dedup by giving each donor a distinct mutated hit string.
# MULTICHAIN_MASK=0 + TMPL_KEEP_FRAME=1 to keep real inter-chain geometry, as in every arm today.
#
# The run ASSERTS "Found 5 templates"; a silently-degraded multi-template arm produces perfectly
# plausible poses and has fooled this project before.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-0}
ROUNDS=${ROUNDS:-3}
SEED=${SEED:-1}            # seed for the templated runs
TIDOFF=${TIDOFF:-0}       # shifts template-id prefixes so reruns cannot collide
P=$REPO/artifacts/pxm2025
M=$REPO/artifacts/${MTAG:-mchain}
mkdir -p "$M"
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) pausing pxm$GPU"
CHUNK=$(tmux capture-pane -t "pxm$GPU" -p 2>/dev/null | grep -E '^### GPU'"$GPU"' start' | tail -1 \
        | awk '{print $4"_"$6}' | sed 's/+.*//')
tmux kill-session -t "pxm$GPU" 2>/dev/null
sleep 10
if [ -n "$CHUNK" ] && [ -d "$P/out/.claims_chunk/$CHUNK" ]; then
  rmdir "$P/out/.claims_chunk/$CHUNK" 2>/dev/null && echo "    released chunk claim $CHUNK"
fi

# round 0 = the five FREE samples of seed 300014 (DockQ 0.683 0.562 0.031 0.033 0.023)
CUR=$(ls $REPO/artifacts/faithful50/g1/9x1h/seed_300014/predictions/9x1h_sample_*.cif | sort)
echo "=== round 0 donors (free seed 300014, all 5 samples):"
echo "$CUR" | sed 's/^/    /'

for r in $(seq 1 "$ROUNDS"); do
  D=$M/r$r
  mkdir -p "$D/out" "$D/pxm_json"
  echo "=== $(date +%F\ %H:%M:%S) round $r: building $(echo "$CUR" | wc -l) templates"
  $PY scripts/build_multipose_template.py --out "$D" --tid-prefix "$(printf '%b' "$(printf '\\x%x' $((0x60+r+TIDOFF)))")" \
      --poses $CUR || { echo "    build FAILED"; break; }

  cd /home/ebrahim/Protenix || exit 1
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
  PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=5 \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$D/job.json" \
    --dump_dir "$D/out" \
    --seeds "$SEED" \
    --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
    --use_template true > "$D/run.log" 2>&1
  cd "$REPO" || exit 1
  echo "    templates found: $(grep -oE 'Found [0-9]+ templates' "$D/run.log" | sort -u | tr '\n' ' ')"
  grep -qE 'Found [0-4] templates' "$D/run.log" && echo "    !! FEWER THAN 5 TEMPLATES -- donors were deduped away"

  NEW=$(find "$D/out" -name '*_sample_*.cif' | sort)
  n=$(echo "$NEW" | grep -c . )
  if [ "$n" -lt 1 ]; then echo "    NO POSES -- stopping"; break; fi
  for f in $NEW; do
    k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
    o=$D/pxm_json/${k}.json
    [ -s "$o" ] || envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
  done
  $PY - "$D" "$r" <<'EOF'
import glob, json, os, sys
import numpy as np
D, r = sys.argv[1], sys.argv[2]
def cidx(n):
    v=0
    for c in n.split("-")[0]: v=v*26+(ord(c.upper())-64)
    return v-1
dq, ip = [], []
for f in sorted(glob.glob(f"{D}/pxm_json/*.json")):
    k=os.path.basename(f).split(".")[0]
    J=json.load(open(f))
    v=(J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
    if not v or v.get("dockq") is None: continue
    dq.append(float(v["dockq"]))
    r2m=J.get("ref_to_model_chain_mapping",{})
    cf=glob.glob(f"{D}/out/**/*summary_confidence_sample_{k}.json", recursive=True)
    p=float("nan")
    if cf and all(c in r2m for c in ("C","D")):
        M=json.load(open(cf[0])).get("chain_pair_iptm")
        i,j=cidx(r2m["C"]),cidx(r2m["D"])
        if M and 0<=i<len(M) and 0<=j<len(M[i]): p=float(M[i][j])
    ip.append(p)
d=np.array(dq); p=np.array(ip)
print(f"    -> round {r}: n={len(d)}  DockQ mean {d.mean():.3f} min {d.min():.3f} max {d.max():.3f}"
      f"  acc {(d>=0.23).sum()}/{len(d)}  ipTM mean {np.nanmean(p):.3f} max {np.nanmax(p):.3f}")
open(f"{D}/../trajectory.csv","a").write(
    f"{r},{len(d)},{d.mean():.4f},{d.min():.4f},{d.max():.4f},{int((d>=0.23).sum())},"
    f"{np.nanmean(p):.4f},{np.nanmax(p):.4f}\n")
EOF
  CUR=$NEW
done

echo "=== trajectory: round,n,dockq_mean,dockq_min,dockq_max,acc,iptm_mean,iptm_max"
echo "    0,5,0.2664,0.0230,0.6830,2,0.1070,0.1070   (free seed 300014, round-0 donors)"
cat "$M/trajectory.csv" 2>/dev/null | sed 's/^/    /'
echo "=== $(date +%F\ %H:%M:%S) handing GPU$GPU back to PXM"
tmux new-session -d -s "pxm$GPU" -c "$REPO" "GPU=$GPU bash ops/run_pxm2025_chunked.sh"
sleep 4
tmux has-session -t "pxm$GPU" 2>/dev/null && echo "    pxm$GPU up" || echo "    FAILED pxm$GPU"
echo "=== $(date +%F\ %H:%M:%S) DONE"
