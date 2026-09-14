#!/usr/bin/env bash
# Re-run of the 43 protein-protein targets whose interface is between SYMMETRY COPIES (A | A-2).
#
# The first pass built inputs from the RCSB asymmetric unit, which for a homodimer holds only ONE
# copy -- so those targets were predicted as monomers and the benchmark interface does not exist in
# the output. No DockQ is recoverable from them; they must be re-predicted, not re-scored.
# Inputs here come from `protenix json` on the biological assembly (artifacts/pp/jsons_asm), which
# yields the correct copy count. See ops/pp_prep_asm.sh.
#
# SEPARATE ARM by design: its own jsons/out/locks, so the original ASU run is left untouched and
# nothing has to be deleted to make room for this.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
PP=$REPO/artifacts/pp
GPU=$1
LOCKS=$PP/locks_asm
QUEUE=$PP/asm_queue.txt
NEED=${NEED:-250}
mkdir -p "$LOCKS" "$PP/out_asm" "$PP/logs_asm"
cd /home/ebrahim/Protenix || exit 1
n_poses(){ find "$PP/out_asm/$1" -name '*_sample_*.cif' 2>/dev/null | wc -l; }

N_SAMPLE=${N_SAMPLE:-5}
missing_seeds(){
  local pdb=$1 out="" s n
  for s in $(seq 101 150); do
    n=$(find "$PP/out_asm/$pdb" -path "*seed_${s}/predictions/*_sample_*.cif" 2>/dev/null | wc -l)
    [ "$n" -ge "$N_SAMPLE" ] && continue
    out="${out:+$out,}$s"
  done
  printf '%s' "$out"
}

echo "### PP-ASM worker GPU$GPU $(date +%H:%M)"
while true; do
  claimed=""
  for pdb in $(awk '{print $1}' "$QUEUE"); do
    [ -s "$PP/jsons_asm/${pdb}-final-updated.json" ] || continue
    [ "$(n_poses "$pdb")" -ge "$NEED" ] && continue
    if [ -d "$LOCKS/$pdb" ] && [ -n "$(find "$LOCKS/$pdb" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then
      pgrep -f "pp/jsons_asm/${pdb}-final" >/dev/null 2>&1 || { echo "### reclaim stale $pdb"; rmdir "$LOCKS/$pdb" 2>/dev/null; }
    fi
    if mkdir "$LOCKS/$pdb" 2>/dev/null; then claimed="$pdb"; break; fi
  done
  [ -z "$claimed" ] && { echo "### GPU$GPU nothing left, exiting $(date +%H:%M)"; break; }
  # RESUME: only the seeds not already complete -- Protenix has no skip-if-exists, so the full
  # 101-150 list makes a restarted worker redo finished seeds from scratch. See ops/run_pp.sh.
  todo=$(missing_seeds "$claimed")
  [ -z "$todo" ] && { echo "### GPU$GPU $claimed already complete"; continue; }
  echo "### GPU$GPU claimed $claimed ($(n_poses "$claimed")/$NEED, seeds to run: $(echo $todo | tr ',' ' ' | wc -w)) $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold $PY runner/inference.py \
      --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$PP/jsons_asm/${claimed}-final-updated.json" \
      --dump_dir "$PP/out_asm/$claimed" \
      --seeds $todo \
      --sample_diffusion.N_sample 5 \
      --sample_diffusion.N_step 200 \
      --use_template true >> "$PP/logs_asm/${claimed}.log" 2>&1 \
    && echo "### GPU$GPU DONE $claimed poses=$(n_poses "$claimed") $(date +%H:%M)" \
    || echo "### GPU$GPU FAIL $claimed $(date +%H:%M)"
done
