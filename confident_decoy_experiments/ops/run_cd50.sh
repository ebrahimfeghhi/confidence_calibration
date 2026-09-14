#!/bin/bash
# 50 more seeds for every confident-decoy complex.
#
# Confident decoy = the stock 50-seed pool contains an acceptable pose (oracle DockQ >= 0.23) but
# pooled-ipTM selection picks one below 0.23. 26 such interfaces over 20 complexes, from
# artifacts/abag_selection_protocol.csv.
#
# Seed band 200000-200049: every existing pool for these complexes tops out below that (the highest
# is 8che at 99899), so these are genuinely new seeds and pose_index will not double-count them.
#
# Claim-based so more GPUs can be added later: each worker mkdirs a claim dir and skips complexes
# already claimed. Resumes -- a complex whose output already holds 50*N_SAMPLE cifs is skipped.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT=$REPO/artifacts/sweep_out/cd50more
CLAIM=$OUT/.claims
GPU=${GPU:-6}
N_SAMPLE=${N_SAMPLE:-5}
SEEDS=$(seq -s, 200000 200049)
mkdir -p "$CLAIM"

PDBS="7ucx 7zau 8aok 8av2 8be1 8bf4 8che 8fas 8fbw 8ivx 8oxw 8oxx 8pn0 8q3j 8slb 8the 8ux6 8vui 8yxi 9jbq"

# PER-SEED resume. Without this, stopping a worker mid-complex throws away every finished seed of
# that complex (the old code only skipped complexes that were 100% done), which cost ~5 h of 7zau
# when PXM-2025 took priority. Recomputed at launch so an interrupted complex restarts where it left
# off.
missing_seeds(){
  local pdb=$1 out="" s n
  for s in $(seq 200000 200049); do
    n=$(find "$OUT/$pdb" -path "*seed_${s}/predictions/*_sample_*.cif" 2>/dev/null | wc -l)
    [ "$n" -ge "$N_SAMPLE" ] && continue
    out="${out:+$out,}$s"
  done
  printf '%s' "$out"
}

cd /home/ebrahim/Protenix || exit 1
for pdb in $PDBS; do
  mkdir "$CLAIM/$pdb" 2>/dev/null || continue          # someone else has it
  todo=$(missing_seeds "$pdb")
  if [ -z "$todo" ]; then
    echo "### GPU$GPU skip $pdb (all 50 seeds present)"; continue
  fi
  echo "### GPU$GPU start $pdb  $(date +%H:%M)  seeds remaining: $(echo $todo | tr ',' '\n' | wc -l)"
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
  $PY runner/inference.py --model_name protenix-v2 \
    --load_checkpoint_dir /home/ebrahim/checkpoint \
    --input_json_path /home/ebrahim/Protenix/foldbench_abag/jsons/${pdb}-final-updated.json \
    --dump_dir "$OUT/$pdb" \
    --seeds "$todo" \
    --sample_diffusion.N_sample $N_SAMPLE --sample_diffusion.N_step 200 \
    --use_template true >> "$OUT/${pdb}.log" 2>&1
  echo "### GPU$GPU done $pdb $(date +%H:%M) cifs=$(find "$OUT/$pdb" -name '*_sample_*.cif' | wc -l)"
done
echo "### GPU$GPU ALL DONE $(date +%H:%M)"
