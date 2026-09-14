#!/usr/bin/env bash
# Does predicting ONE copy of 9x1h instead of two change the docking behaviour?
#
# 9x1h is deposited as two copies of the same antibody-antigen complex -- a crystal packing artifact,
# not biology. Protenix therefore builds both, and they come out near-identical: chain-to-chain CA
# RMSD 0.11-0.53 A, whole-protomer 0.50 A, and the two protomers of one pose scored DockQ 0.643 vs
# 0.634. So ~4x the pair-tensor compute (1474 tokens -> 737 quarters the N^2 pair rep) is being spent
# to produce a duplicate answer.
#
# WHAT COULD STILL GO WRONG, and why this needs testing rather than just switching: the model sees all
# six chains at once when it decides where to dock. Removing the second copy might change the first --
# possibly for the better, possibly not. It is also unknown whether the confident decoy basin, which
# is the whole subject of this project, even survives at count=1.
#
# PAIRED DESIGN: seeds 300000-300029 are exactly the first 30 of the 50 the original sweep ran at
# count=2, so every count=1 seed has a count=2 twin generated from the same seed. STOCK CONFIG here
# (no MSA pinning, MC dropout at its 0.4 default) precisely because the sweep was stock -- pinning
# would change two things at once and make the comparison meaningless.
#
# Ca ligands halved 4 -> 2 to keep stoichiometry consistent with one protein copy.
#
# GPU4 by explicit one-off authorisation; the standing rule is a hard maximum of 4 GPUs and
# allpose20 already holds 0/1/2/6. This does not extend to any later run.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
GPU=${GPU:-4}
C=$REPO/artifacts/count1
mkdir -p "$C/out" "$C/pxm_json"

cd /home/ebrahim/Protenix || exit 1
echo "=== $(date +%H:%M:%S) count=1, seeds 300000-300029, 5 samples each, stock config"
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold \
$PY runner/inference.py --model_name protenix-v2 \
  --load_checkpoint_dir /home/ebrahim/checkpoint \
  --input_json_path "$C/job.json" \
  --dump_dir "$C/out" \
  --seeds "$(seq -s, 300000 300029)" \
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200 \
  --use_template true > "$C/run.log" 2>&1
echo "=== $(date +%H:%M:%S) exit $?  poses=$(find "$C/out" -name '*_sample_*.cif' | wc -l) (expect 150)"
grep -oE 'N_asym [0-9]+, N_token [0-9]+, N_atom [0-9]+' "$C/run.log" | head -1 | sed 's/^/    count=1: /'
echo "    count=2 was:   N_asym 10, N_token 1474, N_atom 11350"

cd "$REPO" || exit 1
echo "=== $(date +%H:%M:%S) scoring"
for f in $(find "$C/out" -name '*_sample_*.cif' | sort); do
  s=$(echo "$f" | grep -oE 'seed_[0-9]+' | grep -oE '[0-9]+')
  k=$(echo "$f" | grep -oE 'sample_[0-9]+' | grep -oE '[0-9]+')
  o=$C/pxm_json/${s}_${k}.json
  [ -s "$o" ] || envs/pxmeter/bin/pxm -r artifacts/pxm2025/cif/9x1h.cif -m "$f" -o "$o" >/dev/null 2>&1
done
echo "    scored $(ls $C/pxm_json/*.json 2>/dev/null | wc -l)/150"
$PY scripts/read_count1.py
echo "=== $(date +%H:%M:%S) DONE"
