#!/usr/bin/env bash
# One GPU worker: template on EACH of 9x1h's 250 sweep poses individually, 20 diffusion samples each.
#
# THE EXPERIMENT. 250 independent runs, one per donor pose. Each run templates on that single pose
# with full inter-chain geometry and draws 20 samples. Not recursive -- this isolates what ONE
# templating step does, per donor, with enough samples to characterise the distribution it produces.
#
# WHAT IT CAN ANSWER THAT THE CHAINS COULD NOT. Every label-free selector tried so far has failed on
# this target: ipTM ranks the first good pose 159/250 (worse than chance), and clustering the poses
# then ranking clusters by size, tightness or confidence all ship the same 0.019 decoy. This run adds
# a signal none of those had -- SELF-CONSISTENCY. For each donor we can ask how faithfully the model
# reproduces what it was handed. If good donors are reproduced differently from decoys, that is a
# label-free discriminator, and unlike confidence it is not derived from the model's own beliefs.
#
# SEED = THE DONOR'S OWN SEED, so the five donors from one sweep seed share that seed's MSA draw.
# MSA depth and MC dropout are pinned (full 15828 rows, dropout off) so the donor is the variable.
# NOTE mc_dropout defaults to apply_rate 0.4 in the base config -- it has been ON in every earlier
# run in this project, silently.
#
# MAX_TEMPLATES=1: a single donor. NODEDUP is still required because the self-pose would otherwise
# dedup against the query by mmcif sequence and log "Found 1 templates" for the wrong reason.
# The run asserts a template was found; an untemplated run still emits plausible poses.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:?}
NS=${NS:-20}
P=$REPO/artifacts/pxm2025
M=$REPO/artifacts/allpose20
CSV=$M/results.csv
mkdir -p "$M/.claims" "$M/.done" "$M/logs"
cd "$REPO" || exit 1
[ -s "$CSV" ] || echo "donor_seed,donor_sample,donor_dockq,sample,dockq,iptm" > "$CSV"

for S in $(seq 300000 300049); do
  for K in 0 1 2 3 4; do
    U="${S}_${K}"
    [ -e "$M/.done/$U" ] && continue
    mkdir "$M/.claims/$U" 2>/dev/null || continue
    echo "$GPU" > "$M/.claims/$U/owner"
    DONOR="$P/out/9x1h/9x1h/seed_$S/predictions/9x1h_sample_$K.cif"
    [ -f "$DONOR" ] || { echo "missing $DONOR" >> "$M/logs/gpu$GPU.log"; touch "$M/.done/$U"; continue; }
    D=$M/u$U
    mkdir -p "$D/out" "$D/pxm_json"
    TID=$($PY -c "import json;print(json.load(open('$M/tids.json'))['$U'])")
    echo "=== $(date +%H:%M:%S) GPU$GPU donor $U tid $TID" >> "$M/logs/gpu$GPU.log"

    $PY scripts/build_multipose_template.py --out "$D" --tid-prefix u --tids "$TID" \
        --no-cache-write --poses "$DONOR" >> "$M/logs/gpu$GPU.log" 2>&1 || {
        echo "   build failed $U" >> "$M/logs/gpu$GPU.log"; touch "$M/.done/$U"; continue; }

    # WAIT for room rather than OOM. On 2026-09-10 a neighbour took 73.5 GB on our cards; every
    # inference then died with CUDA OOM, and because the loop marked donors done regardless, 36
    # donors were consumed producing nothing.
    while :; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
      [ "${free:-0}" -ge 50000 ] && break
      echo "   GPU$GPU only ${free}MiB free, waiting" >> "$M/logs/gpu$GPU.log"
      sleep 120
    done
    cd /home/ebrahim/Protenix || exit 1
    CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
    PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
    PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1 \
    PROTENIX_MSA_NROWS=999999 PROTENIX_MSA_STRATEGY=topk \
    $PY runner/inference.py --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$D/job.json" \
      --dump_dir "$D/out" \
      --seeds "$S" \
      --sample_diffusion.N_sample "$NS" --sample_diffusion.N_step 200 \
      --mc_dropout_apply_rate 0 \
      --use_template true > "$D/run.log" 2>&1
    cd "$REPO" || exit 1
    grep -q 'Found 0 templates' "$D/run.log" && \
      echo "   !! $U ran UNTEMPLATED" >> "$M/logs/gpu$GPU.log"

    for f in $(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
      k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
      o=$D/pxm_json/${k}.json
      [ -s "$o" ] || envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
    done
    $PY - "$D" "$S" "$K" "$CSV" <<'EOF' >> "$M/logs/gpu$GPU.log" 2>&1
import csv, fcntl, glob, json, os, sys
import numpy as np
D, S, K, CSV = sys.argv[1:5]
def cidx(n):
    v = 0
    for c in n.split("-")[0]: v = v*26 + (ord(c.upper())-64)
    return v-1
# the donor's own DockQ, from the sweep, so each row carries what it was templated on
dq0 = ""
for r in csv.DictReader(open("artifacts/pxm2025/dockq_pxm2025.csv")):
    if (r["pdb"]=="9x1h" and r["seed"]==S and r["sample"]==K
            and "".join(sorted((r["chain1"],r["chain2"])))=="CD" and r["dockq"]):
        dq0 = f"{float(r['dockq']):.4f}"; break
rows, dd = [], []
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
    rows.append(f"{S},{K},{dq0},{k},{float(v['dockq']):.4f},{p:.4f}")
    dd.append(float(v["dockq"]))
with open(CSV, "a") as fh:                      # four workers append to one file
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.write("\n".join(rows) + "\n")
    fcntl.flock(fh, fcntl.LOCK_UN)
a = np.array(dd)
print(f"   donor {S}_{K} (own DockQ {dq0}): n={len(a)} out mean {a.mean():.3f} "
      f"max {a.max():.3f} acc {(a>=0.23).sum()}/{len(a)}")
EOF
    rm -f "/home/ebrahim/mmcif/$TID.cif"
    # mark done ONLY if this donor produced rows; otherwise release the claim so it is retried
    if [ "$(grep -c "^${S},${K}," "$CSV" 2>/dev/null)" -gt 0 ]; then
      touch "$M/.done/$U"
    else
      echo "   $U produced NO rows -- releasing claim for retry" >> "$M/logs/gpu$GPU.log"
      rm -f "$M/.claims/$U/owner"; rmdir "$M/.claims/$U" 2>/dev/null
    fi
  done
done
echo "=== $(date +%H:%M:%S) GPU$GPU: no donors left" >> "$M/logs/gpu$GPU.log"
