#!/usr/bin/env bash
# After dump50: (1) template on 9x1h's best pose and dump that trunk state, (2) hand the GPUs back to PXM.
#
# THE COMPARISON THIS COMPLETES. Free-run dumps now exist for a good-basin seed (300038, DockQ 0.615)
# and for the confident decoy (300018, DockQ 0.019, ipTM 0.700). Templating on the best pose and
# dumping gives a third trunk state: what the pair rep looks like when the model is HANDED the right
# answer, against finding it alone, against being confidently wrong.
#
# TEMPLATE FLAGS ARE LOAD-BEARING -- both are needed or the experiment silently degrades to
# "template the folds, throw away the dock":
#   PROTENIX_MULTICHAIN_MASK=0   otherwise every cross-chain template feature is zeroed (hardcoded on
#                                in training, no config knob)
#   PROTENIX_TMPL_KEEP_FRAME=1   otherwise each chain is centred on its own centroid, superimposing all
#                                chains of the template at the origin
#   PROTENIX_TMPL_NODEDUP=1 + PROTENIX_MAX_TEMPLATES=1   the dedup keys on mmcif sequence, so a
#                                self-pose collapses against the query and logs "Found 1 templates"
# The run ASSERTS 4 template hits per chain were found before it is trusted.
#
# 5 SEEDS, ALL DUMPED AND ALL KEPT (~16.5 GB): unlike the hunt, every seed here is informative -- the
# question is what templating does to the trunk, not whether a seed gets lucky. Seeds 1-5 rather than
# the 300000 band so these cannot be confused with free-run seeds.
#
# cwd=/home/ebrahim/Protenix and a relative runner path: the only configuration observed to reach the
# good basin today (3/36 vs 0/125 elsewhere). Why is unexplained -- see the cwd memory -- so the
# working recipe is copied exactly rather than improvised.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
P=$REPO/artifacts/pxm2025
T=$REPO/artifacts/tmplbest
cd "$REPO" || exit 1

echo "=== $(date +%F\ %H:%M:%S) waiting for dump50"
while tmux ls 2>/dev/null | grep -qE '^d[0126]:'; do sleep 60; done
echo "=== $(date +%F\ %H:%M:%S) dump50 finished"
echo "    seeds: $(grep -hcE 'max DockQ' artifacts/dump50/driver_g*.log 2>/dev/null | paste -sd+ | bc)/50"
grep -hE '\-> HIT' artifacts/dump50/driver_g*.log 2>/dev/null | sed 's/^/      /'
echo "    free-run dumps kept:"; ls -lh artifacts/dump50/reps 2>/dev/null | awk 'NR>1{print "      "$9"  "$5}'

echo "=== $(date +%F\ %H:%M:%S) building the best-pose template"
$PY scripts/build_bestpose_template.py --out "$T" || { echo "    template build FAILED"; exit 1; }

echo "=== $(date +%F\ %H:%M:%S) templated run, 5 seeds, DUMP_FULL on"
mkdir -p "$T/out" "$T/reps"
cd /home/ebrahim/Protenix || exit 1
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold \
PROTENIX_MULTICHAIN_MASK=0 PROTENIX_TMPL_KEEP_FRAME=1 \
PROTENIX_TMPL_NODEDUP=1 PROTENIX_MAX_TEMPLATES=1 \
DUMP_FULL=1 DUMP_FULL_DIR="$T/reps" \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$T/job.json" \
  --dump_dir "$T/out" \
  --seeds 1,2,3,4,5 \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$T/run.log" 2>&1
echo "    exit $?  poses=$(find "$T/out" -name '*_sample_*.cif' 2>/dev/null | wc -l) (expect 25)"

# a templated arm that found no templates is worthless and must not be read as a result
ft=$(grep -c 'Found [0-9]* templates' "$T/run.log" 2>/dev/null)
echo "    template lines: $ft"; grep -m3 -oE 'Found [0-9]+ templates' "$T/run.log" | sed 's/^/      /'
if grep -q 'Found 0 templates' "$T/run.log" 2>/dev/null; then
  echo "    !! FOUND 0 TEMPLATES -- this arm ran untemplated; treat results as void"
fi

cd "$REPO" || exit 1
echo "=== $(date +%F\ %H:%M:%S) scoring the templated poses"
mkdir -p "$T/pxm_json"
for f in $(find "$T/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
  s=$(echo "$f" | grep -oE 'seed_[0-9]+' | grep -oE '[0-9]+')
  k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
  o=$T/pxm_json/${s}_${k}.json
  [ -s "$o" ] || envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
done
$PY - <<'EOF'
import glob, json, os, re, numpy as np
T="/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/tmplbest"
ch=json.load(open(f"{T}/chosen.json"))
print(f"    templated on: {ch['arm']} seed {ch['seed']} sample {ch['sample']}, donor CD DockQ {ch['dockq']:.3f}")
rows=[]
for f in sorted(glob.glob(f"{T}/pxm_json/*.json")):
    m=re.match(r"(\d+)_(\d+)\.json$", os.path.basename(f))
    try: J=json.load(open(f))
    except Exception: continue
    v=(J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
    if v and v.get("dockq") is not None: rows.append((int(m.group(1)),int(m.group(2)),float(v["dockq"])))
if rows:
    d=np.array([r[2] for r in rows])
    print(f"    templated poses: n={len(d)} max {d.max():.3f} mean {d.mean():.3f} "
          f"acceptable {(d>=0.23).sum()}/{len(d)}")
EOF
echo "    trunk dumps:"; ls -lh "$T/reps" 2>/dev/null | awk 'NR>1{print "      "$9"  "$5}'
echo "    quota: $(quota -s 2>/dev/null | tail -1 | awk '{print $2" of "$4}')"


echo "=== $(date +%F\ %H:%M:%S) STAGE 3: 5 free seeds, dumped and ALL KEPT, to build a decoy set"
# WHY. We now hold three good-basin trunk dumps (300038/300048/300022) but only two decoy dumps
# (300018 ipTM 0.700, and the dump9x1h 300014 run at ipTM 0.506). Comparing one basin against the
# other needs several of each, or "what the good basin looks like" cannot be told apart from "what
# seed 300038 looks like".
# NOTHING IS PRUNED HERE: unlike the hunt, a miss IS the object of study. On this target 58% of seeds
# come back committed (ipTM >= 0.35), so 5 seeds should yield ~3 confident decoys.
# Fresh seed band 300050+ so these can never be confused with the 300000-049 free-run pool.
D=$REPO/artifacts/decoys
mkdir -p "$D/out" "$D/reps" "$D/pxm_json"
cd /home/ebrahim/Protenix || exit 1
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold \
DUMP_FULL=1 DUMP_FULL_DIR="$D/reps" \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$P/jsons/9x1h-final-updated.json" \
  --dump_dir "$D/out" \
  --seeds 300050,300051,300052,300053,300054 \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$D/run.log" 2>&1
echo "    exit $?  poses=$(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | wc -l) (expect 25)"
cd "$REPO" || exit 1
for f in $(find "$D/out" -name '*_sample_*.cif' 2>/dev/null | sort); do
  s=$(echo "$f" | grep -oE 'seed_[0-9]+' | grep -oE '[0-9]+')
  k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
  o=$D/pxm_json/${s}_${k}.json
  [ -s "$o" ] || envs/pxmeter/bin/pxm -r "$P/cif/9x1h.cif" -m "$f" -o "$o" >/dev/null 2>&1
done
$PY - <<'EOF'
import glob, json, os, re, numpy as np
D="/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/decoys"
def cidx(n):
    v=0
    for c in n.split("-")[0]: v=v*26+(ord(c.upper())-64)
    return v-1
by={}
for f in sorted(glob.glob(f"{D}/pxm_json/*.json")):
    m=re.match(r"(\d+)_(\d+)\.json$", os.path.basename(f))
    try: J=json.load(open(f))
    except Exception: continue
    v=(J.get("interface") or {}).get("C,D") or (J.get("interface") or {}).get("D,C")
    if not v or v.get("dockq") is None: continue
    r2m=J.get("ref_to_model_chain_mapping",{})
    cf=f"{D}/out/9x1h/seed_{m.group(1)}/predictions/9x1h_summary_confidence_sample_{m.group(2)}.json"
    ip=float("nan")
    if os.path.exists(cf) and all(c in r2m for c in ("C","D")):
        M=json.load(open(cf)).get("chain_pair_iptm")
        i,j=cidx(r2m["C"]),cidx(r2m["D"])
        if M and 0<=i<len(M) and 0<=j<len(M[i]): ip=float(M[i][j])
    by.setdefault(int(m.group(1)),[]).append((float(v["dockq"]),ip))
print(f"    {'seed':>8} {'maxDockQ':>9} {'meanipTM':>9}  state")
for s in sorted(by):
    d=np.array([x[0] for x in by[s]]); ip=np.nanmean([x[1] for x in by[s]])
    st="GOOD" if d.max()>=0.23 else ("committed decoy" if ip>=0.35 else "uncommitted")
    print(f"    {s:>8} {d.max():>9.3f} {ip:>9.3f}  {st}")
EOF
echo "    decoy dumps: $(ls "$D/reps" 2>/dev/null | wc -l) files, $(du -x -d0 -BG "$D/reps" 2>/dev/null | cut -f1)"

echo "=== $(date +%F\ %H:%M:%S) releasing abandoned chunk claims"
for c in $(ls "$P/out/.claims_chunk" 2>/dev/null); do
  pdb=${c%_*}; st=${c##*_}
  n=$(ls "$P/out/$pdb/$pdb/seed_$st/predictions/"*_sample_*.cif 2>/dev/null | wc -l)
  [ "$n" -lt 5 ] && rmdir "$P/out/.claims_chunk/$c" 2>/dev/null && echo "    released $c ($n poses)"
done
echo "=== $(date +%F\ %H:%M:%S) restarting PXM on GPUs 0 1 2 6"
for g in 0 1 2 6; do
  tmux has-session -t "pxm$g" 2>/dev/null && { echo "    pxm$g already up"; continue; }
  tmux new-session -d -s "pxm$g" -c "$REPO" "GPU=$g bash ops/run_pxm2025_chunked.sh"
  sleep 3
  tmux has-session -t "pxm$g" 2>/dev/null && echo "    pxm$g up" || echo "    FAILED pxm$g"
done
echo "=== $(date +%F\ %H:%M:%S) ALL DONE"
