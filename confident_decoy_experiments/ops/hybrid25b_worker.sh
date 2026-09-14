#!/usr/bin/env bash
# Matched-compute test: does "25 free seeds + 25 self-templated" beat "50 free seeds"?
#
# THE COMPARISON. The PXM-2025 sweep already ran 50 free seeds x 5 samples = 250 poses per complex.
# This generates the other arm: for each of the first 25 seeds, template on ALL FIVE of that seed's
# poses at once and run under seed i+25 with 5 samples. Arm B is then 25 free + 25 templated = 250
# poses, the SAME compute as arm A. Both are scored by what ipTM actually selects.
#
# STOCK CONFIG IS DELIBERATE -- MSA subsampling on, mc_dropout at its 0.4 default. The free arm was
# generated that way, and pinning them here would change two variables at once and void the
# comparison. (Note mc_dropout has been on by default in every run of this project; protenix.py:728
# does mc_dropout = random.random() < apply_rate.)
#
# 60 complexes x 3 classes: 30 confident decoys (ipTM ships a bad pose though a good one exists),
# 15 no-coverage (no acceptable pose in 50 seeds -- a control, since there is nothing to amplify),
# 15 solved. Units run EASY FIRST: single-copy assemblies before multi-copy, smallest first, so the
# tricky chain assignments run last and can be checked while the simple ones occupy the GPUs.
#
# TEMPLATE FLAGS are load-bearing: MULTICHAIN_MASK=0 keeps cross-chain template features (otherwise
# the dock is discarded), TMPL_KEEP_FRAME=1 stops per-chain centring collapsing the template onto the
# origin, TMPL_NODEDUP=1 + MAX_TEMPLATES=5 stop five self-poses being deduped to one.
#
# GPU MEMORY: waits for room rather than OOMing. Earlier today a neighbour filled our cards and every
# inference died; because the loop marked units done regardless, 36 units were consumed producing
# nothing. Units are now marked done only after rows are written.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
P=$REPO/artifacts/pxm2025
M=$REPO/artifacts/hybrid25b
CSV=$M/results.csv
mkdir -p "$M/.claims" "$M/.done" "$M/logs"
cd "$REPO" || exit 1
[ -s "$CSV" ] || echo "pdb,interface,cls,donor_seed,new_seed,sample,dockq,iptm" > "$CSV"

tail -n +2 "$M/units.csv" | while IFS=, read -r pdb interface cls tokens multi dseed nseed; do
  U="${pdb}_${dseed}"
  [ -e "$M/.done/$U" ] && continue
  mkdir "$M/.claims/$U" 2>/dev/null || continue
  echo "$GPU" > "$M/.claims/$U/owner"
  LOG=$M/logs/gpu$GPU.log
  POSES=$(ls "$P/out/$pdb/$pdb/seed_$dseed/predictions/${pdb}_sample_"*.cif 2>/dev/null | sort)
  if [ "$(echo "$POSES" | grep -c .)" -lt 5 ]; then
    echo "$(date +%H:%M:%S) $U: donor poses missing, skipping" >> "$LOG"
    touch "$M/.done/$U"; continue
  fi
  D=$M/$pdb/s$dseed
  mkdir -p "$D/out" "$D/pxm_json"
  TIDS=$($PY -c "
import json;t=json.load(open('$M/tids.json'))
print(' '.join(t[f'${pdb}_${dseed}_{d}'] for d in range(5)))")
  echo "$(date +%H:%M:%S) GPU$GPU $U ($cls, $tokens tok, multi=$multi)" >> "$LOG"
  $PY scripts/build_selftmpl_general.py --pdb "$pdb" --out "$D" --poses $POSES \
      --tids $TIDS --no-cache-write >> "$LOG" 2>&1 || {
      echo "   build failed $U" >> "$LOG"; rm -rf "$M/.claims/$U"; continue; }

  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
    [ "${free:-0}" -ge 45000 ] && break
    echo "   GPU$GPU ${free}MiB free, waiting" >> "$LOG"; sleep 120
  done

  cd /home/ebrahim/Protenix || exit 1
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
  PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=5 \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$D/job.json" --dump_dir "$D/out" \
    --seeds "$nseed" --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
    --use_template true > "$D/run.log" 2>&1
  cd "$REPO" || exit 1
  grep -qE 'Found [0-4] templates' "$D/run.log" && echo "   $U: FEWER THAN 5 TEMPLATES" >> "$LOG"

  for f in $(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
    k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
    o=$D/pxm_json/${k}.json
    [ -s "$o" ] || envs/pxmeter/bin/pxm -r "$P/cif/$pdb.cif" -m "$f" -o "$o" >/dev/null 2>&1
  done
  $PY - "$D" "$pdb" "$interface" "$cls" "$dseed" "$nseed" "$CSV" <<'EOF' >> "$LOG" 2>&1
import fcntl, glob, json, os, sys
import numpy as np
D,pdb,interface,cls,ds,ns,CSV = sys.argv[1:8]
c1,c2 = interface.split("_")[1][0], interface.split("_")[1][1]
def cidx(n):
    v=0
    for c in n.split("-")[0]: v=v*26+(ord(c.upper())-64)
    return v-1
rows,dd=[],[]
for f in sorted(glob.glob(f"{D}/pxm_json/*.json")):
    k=os.path.basename(f).split(".")[0]
    try: J=json.load(open(f))
    except Exception: continue
    iface=J.get("interface") or {}
    v=iface.get(f"{c1},{c2}") or iface.get(f"{c2},{c1}")
    if not v or v.get("dockq") is None: continue
    r2m=J.get("ref_to_model_chain_mapping",{})
    cf=glob.glob(f"{D}/out/**/*summary_confidence_sample_{k}.json", recursive=True)
    p=float("nan")
    if cf and all(c in r2m for c in (c1,c2)):
        M=json.load(open(cf[0])).get("chain_pair_iptm")
        i,j=cidx(r2m[c1]),cidx(r2m[c2])
        if M and 0<=i<len(M) and 0<=j<len(M[i]): p=float(M[i][j])
    rows.append(f"{pdb},{interface},{cls},{ds},{ns},{k},{float(v['dockq']):.4f},{p:.4f}")
    dd.append(float(v["dockq"]))
if rows:
    with open(CSV,"a") as fh:
        fcntl.flock(fh,fcntl.LOCK_EX); fh.write("\n".join(rows)+"\n"); fcntl.flock(fh,fcntl.LOCK_UN)
    a=np.array(dd)
    print(f"   -> {pdb} s{ds}: n={len(a)} DockQ mean {a.mean():.3f} max {a.max():.3f} "
          f"acc {(a>=0.23).sum()}/{len(a)}")
else:
    print(f"   -> {pdb} s{ds}: NO ROWS")
EOF
  for t in $TIDS; do rm -f "/home/ebrahim/mmcif/$t.cif"; done
  if [ "$(grep -c "^${pdb},${interface},${cls},${dseed}," "$CSV" 2>/dev/null)" -gt 0 ]; then
    touch "$M/.done/$U"
  else
    echo "   $U produced no rows -- releasing for retry" >> "$LOG"
    rm -rf "$M/.claims/$U"
  fi
done
echo "=== $(date +%H:%M:%S) GPU$GPU: no units left" >> "$M/logs/gpu$GPU.log"
