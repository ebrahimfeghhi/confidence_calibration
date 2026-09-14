#!/usr/bin/env bash
# PXM-2025 ab-ag generation, SECOND BATCH of free seeds (default 300075-300099).
#
# Same generator as run_pxm2025_chunked.sh, only the seed range moves. Output goes into the SAME
# artifacts/pxm2025/out/<pdb> tree, so downstream scoring sees one pool of 75 free seeds per complex
# rather than two arms to reconcile. Chunk claims are keyed <pdb>_<firstseed>, and the new first-seeds
# (300075+) cannot collide with the original ones (300000-300040).
#
# SEEDS 300075-300099 deliberately skip 300050-300074: those are the RUN seeds of the hybrid25b
# templated arm. Different output tree, but reusing the numbers would make the two impossible to tell
# apart in any pooled analysis.
#
# WHY: with complex-level claims the queue ends badly. 8rvr alone is ~16.8 GPU-hours, so at the tail
# one worker grinds it while the other three sit idle -- roughly 10 wasted GPU-hours. Splitting the
# 50 seeds of a complex across workers means four GPUs finish 8rvr in ~4.2 h and nobody idles.
#
# CHUNK SIZE IS ADAPTIVE. Every chunk pays a fresh model load (~1 min), which is noise on a 17-hour
# complex and pure waste on a 7-second-per-seed one. So: cheap complexes stay a single 50-seed chunk,
# expensive ones split into 10-seed chunks. Threshold is on predicted cost, using the cost model
# measured on this run (sec/seed = 0.000244 * N_token^1.86, correlation 0.998 over 12 complexes).
#
# COEXISTS WITH ops/run_pxm2025.sh: a complex already claimed whole under the old scheme
# (.claims/<pdb>) is skipped here, so old and new workers can run side by side without duplicating.
# Chunk claims live in .claims_chunk/<pdb>_<firstseed>.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
P=$REPO/artifacts/pxm2025
OUT=$P/out
CLAIM=$OUT/.claims            # old, whole-complex claims
CCLAIM=$OUT/.claims_chunk     # new, per-chunk claims
GPU=${GPU:-0}
SEED_LO=${SEED_LO:-300075}
NSEEDS=${NSEEDS:-25}
N_SAMPLE=${N_SAMPLE:-5}
BIG_SECONDS=${BIG_SECONDS:-3600}   # predicted complex cost above which we split
mkdir -p "$CCLAIM"

# seeds of $1 in chunk starting at $2 (length $3) that do NOT already have all N_SAMPLE poses
missing_in_chunk(){
  local pdb=$1 start=$2 len=$3 out="" s n
  for s in $(seq "$start" $((start + len - 1))); do
    n=$(find "$OUT/$pdb" -path "*seed_${s}/predictions/*_sample_*.cif" 2>/dev/null | wc -l)
    [ "$n" -ge "$N_SAMPLE" ] && continue
    out="${out:+$out,}$s"
  done
  printf '%s' "$out"
}

cd /home/ebrahim/Protenix || exit 1
while read -r pdb; do
  [ -z "$pdb" ] && continue
  j=$P/jsons/${pdb}-final-updated.json
  [ -s "$j" ] || continue
  [ -d "$CLAIM/$pdb" ] && continue        # whole complex owned by an old-scheme worker

  # predicted cost -> chunk size
  est=$($PY - "$j" "$NSEEDS" <<'EOF'
import json, sys
b = json.load(open(sys.argv[1])); b = b[0] if isinstance(b, list) else b
n = sum(len(v["sequence"]) * int(v.get("count", 1))
        for x in b["sequences"] for _, v in x.items() if "sequence" in v)
print(int(int(sys.argv[2]) * 0.000244 * n ** 1.86))
EOF
)
  if [ "$est" -gt "$BIG_SECONDS" ]; then CH=5; else CH=$NSEEDS; fi

  for start in $(seq $SEED_LO $CH $((SEED_LO + NSEEDS - 1))); do
    mkdir "$CCLAIM/${pdb}_${start}" 2>/dev/null || continue    # someone else has this chunk
    todo=$(missing_in_chunk "$pdb" "$start" "$CH")
    if [ -z "$todo" ]; then
      echo "### GPU$GPU skip $pdb seeds $start+ (already present)"; continue
    fi
    echo "### GPU$GPU start $pdb seeds $start+$CH ($(echo $todo | tr ',' '\n' | wc -l) to do) est_complex=${est}s $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
    $PY runner/inference.py --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$j" \
      --dump_dir "$OUT/$pdb" \
      --seeds "$todo" \
      --sample_diffusion.N_sample $N_SAMPLE --sample_diffusion.N_step 200 \
      --use_template true >> "$OUT/${pdb}.log" 2>&1
    echo "### GPU$GPU done $pdb seeds $start+$CH $(date +%H:%M) cifs=$(find "$OUT/$pdb" -name '*_sample_*.cif' 2>/dev/null | wc -l)"
  done
done < "$P/order.txt"
echo "### GPU$GPU ALL DONE $(date +%H:%M)"
