#!/usr/bin/env bash
# FoldBench PROTEIN-PROTEIN seed-scaling sweep: 50 representative complexes x 50 seeds x 5 samples.
#
# PURPOSE: the "seed scaling helps ab-ag but not other interfaces" contrast is currently CITED from
# AF3/ESMFold2/Protenix-v2, not measured here. This measures it on the same model, benchmark family
# and protocol as our ab-ag numbers, so the comparison is like-for-like -- and lets us run the same
# mechanism analyses (fnat vs iRMSD/LRMSD, epitope clustering) on a class where scaling is reported
# NOT to help.
#
# Inputs built with Protenix's OWN tooling, verified to reproduce foldbench_abag/jsons byte-for-byte:
#   RCSB asymmetric unit -> `protenix json` (applies CRYSTALLIZATION_AIDS, merges glycan bonds)
#                        -> `protenix mt`   (MSA + template search) -> <pdb>-final-updated.json
# Inference flags copied from /home/ebrahim/Protenix/run_benchmark.sh (the ab-ag benchmark driver).
#
# Work-stealing via atomic mkdir claims: GPUs take non-overlapping complexes, whoever frees first
# takes the next. Smallest-first so results bank early.
set -o pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
PP=$A/pp
GPU=$1
LOCKS=$PP/locks
NEED=${NEED:-250}
mkdir -p "$LOCKS" "$PP/out" "$PP/logs"
cd /home/ebrahim/Protenix || exit 1
n_poses(){ find "$PP/out/$1" -name '*_sample_*.cif' 2>/dev/null | wc -l; }

# Seeds 101-150 that do NOT yet have all N_SAMPLE poses on disk, comma separated.
N_SAMPLE=${N_SAMPLE:-5}
missing_seeds(){
  local pdb=$1 out="" s n
  for s in $(seq 101 150); do
    n=$(find "$PP/out/$pdb" -path "*seed_${s}/predictions/*_sample_*.cif" 2>/dev/null | wc -l)
    [ "$n" -ge "$N_SAMPLE" ] && continue
    out="${out:+$out,}$s"
  done
  printf '%s' "$out"
}

echo "### PP worker GPU$GPU $(date +%H:%M)"
while true; do
  claimed=""
  for pdb in $(awk '{print $1}' "$PP/subset50.txt"); do
    [ "$(n_poses "$pdb")" -ge "$NEED" ] && continue
    if [ -d "$LOCKS/$pdb" ] && [ -n "$(find "$LOCKS/$pdb" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then
      pgrep -f "pp/jsons/${pdb}-final" >/dev/null 2>&1 || { echo "### reclaim stale $pdb"; rmdir "$LOCKS/$pdb" 2>/dev/null; }
    fi
    if mkdir "$LOCKS/$pdb" 2>/dev/null; then claimed="$pdb"; break; fi
  done
  [ -z "$claimed" ] && { echo "### GPU$GPU nothing left, exiting $(date +%H:%M)"; break; }
  # RESUME: pass only the seeds that are not already complete. Protenix has no skip-if-exists
  # logic, so passing the full 101-150 list makes a restarted worker redo every seed from scratch --
  # pausing a GPU at 195/250 poses used to throw away ~4 h of finished work. A seed counts as done
  # when it has all N_SAMPLE poses on disk.
  todo=$(missing_seeds "$claimed")
  [ -z "$todo" ] && { echo "### GPU$GPU $claimed already complete"; continue; }
  echo "### GPU$GPU claimed $claimed ($(n_poses "$claimed")/$NEED, seeds to run: $(echo $todo | tr ',' ' ' | wc -w)) $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold $PY runner/inference.py \
      --model_name protenix-v2 \
      --load_checkpoint_dir /home/ebrahim/checkpoint \
      --input_json_path "$PP/jsons/${claimed}-final-updated.json" \
      --dump_dir "$PP/out/$claimed" \
      --seeds $todo \
      --sample_diffusion.N_sample 5 \
      --sample_diffusion.N_step 200 \
      --use_template true >> "$PP/logs/${claimed}.log" 2>&1 \
    && echo "### GPU$GPU DONE $claimed poses=$(n_poses "$claimed") $(date +%H:%M)" \
    || echo "### GPU$GPU FAIL $claimed $(date +%H:%M)"
done
