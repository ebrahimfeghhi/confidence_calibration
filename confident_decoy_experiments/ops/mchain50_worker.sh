#!/usr/bin/env bash
# One GPU worker for the 50-seed recursive multi-pose self-templating run.
#
# THE EXPERIMENT. For each of 9x1h's 50 original sweep seeds: take that seed's 5 free poses, template
# on ALL FIVE at once, generate 5 new poses, feed those back. 10 rounds. No selection at any point.
#
# WHAT IT SETTLES THAT THE PILOTS COULD NOT. Single chains showed self-templating is a basin
# amplifier -- good donors gave 5/5 acceptable every round, decoy donors 0/5 pinned at 0.018 -- but
# their ipTM readings were useless: across five pilot chains the good basin spanned 0.509-0.756 and
# the decoy 0.316-0.743, fully overlapping, because ipTM is ~87% a between-seed property. Fifty
# chains average that seed term out. Ten rounds also tests the slow leak seen in the pilots: good
# chains drifted DOWN each round (0.792 -> 0.736 -> 0.715) while the decoy sat perfectly still.
#
# WORK CLAIMING, not slicing: seeds are claimed by atomic mkdir with work-stealing, so a slow seed
# cannot strand a GPU and a dead worker's seed can be reclaimed. Static per-GPU slices have stranded
# work in this project before.
#
# TEMPLATE IDS come from artifacts/mchain50/tids.json, pre-registered by scripts/prereg_tids.py, and
# the builder runs with --no-cache-write. Four workers rewriting the 228k-entry release_date_cache
# concurrently would corrupt it, and every job on this box -- including the PXM sweep -- reads it.
#
# EVERY ROUND ASSERTS "Found 5 templates". A silently deduped multi-template arm still emits
# plausible poses; that failure has gone unnoticed in this project more than once.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
ROUNDS=${ROUNDS:-10}
P=$REPO/artifacts/pxm2025
M=$REPO/artifacts/mchain50
CSV=$M/results.csv
mkdir -p "$M/.claims" "$M/.done" "$M/logs"
cd "$REPO" || exit 1

[ -s "$CSV" ] || echo "seed,round,sample,dockq,iptm" > "$CSV"

for S in $(seq 300000 300049); do
  [ -e "$M/.done/$S" ] && continue
  mkdir "$M/.claims/$S" 2>/dev/null || continue     # someone else owns it
  echo "$GPU" > "$M/.claims/$S/owner"
  LOG=$M/logs/seed$S.log
  echo "=== $(date +%F\ %H:%M:%S) GPU$GPU claimed seed $S" | tee -a "$LOG"

  CUR=$(ls "$P/out/9x1h/9x1h/seed_$S/predictions/9x1h_sample_"*.cif 2>/dev/null | sort)
  if [ -z "$CUR" ]; then
    echo "    no round-0 poses for seed $S -- skipping" | tee -a "$LOG"
    touch "$M/.done/$S"; continue
  fi

  ok=1
  for r in $(seq 1 "$ROUNDS"); do
    D=$M/s$S/r$r
    mkdir -p "$D/out" "$D/pxm_json"
    TIDS=$($PY -c "
import json;t=json.load(open('$M/tids.json'))
print(' '.join(t[f'${S}_${r}_{d}'] for d in range(5)))")
    $PY scripts/build_multipose_template.py --out "$D" --tid-prefix w --tids $TIDS \
        --no-cache-write --poses $CUR >> "$LOG" 2>&1 || { echo "    build failed r$r" >>"$LOG"; ok=0; break; }

    cd /home/ebrahim/Protenix || exit 1
    CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
    PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
    PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=5 \
    $PY runner/inference.py --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$D/job.json" \
      --dump_dir "$D/out" \
      --seeds "$S" \
      --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
      --use_template true > "$D/run.log" 2>&1
    cd "$REPO" || exit 1

    if grep -qE 'Found [0-4] templates' "$D/run.log"; then
      echo "    seed $S r$r: FEWER THAN 5 TEMPLATES -- aborting this seed" | tee -a "$LOG"; ok=0; break
    fi
    NEW=$(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | sort)
    [ -z "$NEW" ] && { echo "    seed $S r$r: no poses" | tee -a "$LOG"; ok=0; break; }

    for f in $NEW; do
      k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
      o=$D/pxm_json/${k}.json
      [ -s "$o" ] || envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
    done
    $PY - "$D" "$S" "$r" "$CSV" <<'EOF' >> "$LOG" 2>&1
import fcntl, glob, json, os, sys
import numpy as np
D, S, r, CSV = sys.argv[1:5]
def cidx(n):
    v = 0
    for c in n.split("-")[0]: v = v*26 + (ord(c.upper())-64)
    return v-1
rows, dq = [], []
for f in sorted(glob.glob(f"{D}/pxm_json/*.json")):
    k = os.path.basename(f).split(".")[0]
    try: J = json.load(open(f))
    except Exception: continue
    v = (J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
    if not v or v.get("dockq") is None: continue
    r2m = J.get("ref_to_model_chain_mapping", {})
    cf = glob.glob(f"{D}/out/**/*summary_confidence_sample_{k}.json", recursive=True)
    p = float("nan")
    if cf and all(c in r2m for c in ("C","D")):
        M = json.load(open(cf[0])).get("chain_pair_iptm")
        i, j = cidx(r2m["C"]), cidx(r2m["D"])
        if M and 0 <= i < len(M) and 0 <= j < len(M[i]): p = float(M[i][j])
    rows.append(f"{S},{r},{k},{float(v['dockq']):.4f},{p:.4f}")
    dq.append(float(v["dockq"]))
# flock: four workers append to one csv; without it lines interleave and rows are lost
with open(CSV, "a") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.write("\n".join(rows) + "\n")
    fcntl.flock(fh, fcntl.LOCK_UN)
d = np.array(dq)
print(f"    seed {S} r{r}: n={len(d)} DockQ mean {d.mean():.3f} acc {(d>=0.23).sum()}/{len(d)}")
EOF
    CUR=$NEW
  done

  # reclaim the ~17 MB of template cifs for this seed; 50 seeds would otherwise leave ~875 MB
  for t in $($PY -c "
import json;t=json.load(open('$M/tids.json'))
print(' '.join(v for k,v in t.items() if k.startswith('${S}_')))"); do
    rm -f "/home/ebrahim/mmcif/$t.cif"
  done
  touch "$M/.done/$S"
  echo "=== $(date +%F\ %H:%M:%S) GPU$GPU finished seed $S (ok=$ok)" | tee -a "$LOG"
done
echo "=== $(date +%F\ %H:%M:%S) GPU$GPU: no seeds left"
