#!/usr/bin/env bash
# PXM-2025 antibody-antigen: 50 seeds x 5 diffusion samples for each of the 131 complexes.
#
# WHY: our seed-scaling conclusions (epitope Jaccard explains r^2=0.82 of the DockQ gain; selected
# DockQ +0.167 ab-ag vs +0.030 PP) all come from FoldBench. PXMeter's PXM-2025 ab-ag set shares ZERO
# PDB entries with FoldBench and is drawn from 2025 depositions, so it is an independent replication
# set on structures released after most training cutoffs. 314 interfaces over 131 complexes; none
# exceeds protenix-v2's 2560-token limit.
#
# Seed band 300000-300049 -- distinct from the ab-ag pools (<=99899) and the cd50more band
# (200000-200049), so pose indexing can never conflate arms.
#
# Estimated ~170 GPU-hours total (power-law fit sec/seed = 4.2e-5 * tokens^2.14 to our own measured
# timings). The top 10 complexes are ~35% of that, so LARGEST-LAST ordering below lets the cheap
# majority land early and keeps all workers busy instead of one worker sitting on an 8-hour target
# while the others idle.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
P=$REPO/artifacts/pxm2025
OUT=$P/out
CLAIM=$OUT/.claims
GPU=${GPU:-0}
N_SAMPLE=${N_SAMPLE:-5}
mkdir -p "$CLAIM"

# smallest first: cheap targets finish early so partial results are usable sooner
ORDER=${ORDER:-$P/order.txt}

missing_seeds(){
  local pdb=$1 out="" s n
  for s in $(seq 300000 300049); do
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
  [ -s "$j" ] || { echo "### GPU$GPU MISSING json $pdb"; continue; }
  mkdir "$CLAIM/$pdb" 2>/dev/null || continue          # another worker has it
  todo=$(missing_seeds "$pdb")
  if [ -z "$todo" ]; then
    echo "### GPU$GPU skip $pdb (all 50 seeds present)"; continue
  fi
  echo "### GPU$GPU start $pdb $(date +%H:%M) seeds=$(echo $todo | tr ',' '\n' | wc -l)"
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path "$j" \
    --dump_dir "$OUT/$pdb" \
    --seeds "$todo" \
    --sample_diffusion.N_sample $N_SAMPLE --sample_diffusion.N_step 200 \
    --use_template true >> "$OUT/${pdb}.log" 2>&1
  echo "### GPU$GPU done $pdb $(date +%H:%M) cifs=$(find "$OUT/$pdb" -name '*_sample_*.cif' 2>/dev/null | wc -l)"
done < "$ORDER"
echo "### GPU$GPU ALL DONE $(date +%H:%M)"
