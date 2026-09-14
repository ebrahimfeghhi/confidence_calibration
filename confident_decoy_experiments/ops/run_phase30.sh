#!/usr/bin/env bash
# THREE-PHASE MSA CURRICULUM over 30 recycles (user's design):
#   cycles  0-9  -- NO MSA          (chemistry + templates only)
#   cycles 10-19 -- INTRA-CHAIN MSA (paired/cross-chain rows dropped)
#   cycles 20-29 -- FULL MSA        (paired rows restored)
# via PROTENIX_MSA_PHASES=10,20 and --model.N_cycle 30.
#
# SMOKE-TESTED before launch: verified exactly 10 "none" / 10 "intra" / 10 "full" cycles, and that
# the intra phase actually drops the paired block (msa_featurizer.py:333 puts paired rows on top;
# the count arrives as prot_pair_num_alignments). Measured paired fraction is 11-48% on most
# complexes -- 8qf4's 1.1% turned out to be the outlier -- so the middle phase is a real intervention.
#
# ORDER: ops/order_phase30.txt -- 9 stratified complexes FIRST (3 that the msaoff5 arm made worse,
# 3 unchanged, 3 better), so the early read spans the previous arm's full behaviour instead of being
# biased by the ascending-size ordering that made every early msaoff5 read look falsely positive.
#
# WORK-STEALING: one complex at a time under an atomic mkdir lock, same as run_msaoff5.sh. GPUs run
# non-overlapping complexes and a GPU that finishes early takes whatever is left, so no card idles
# while another still has a queue.
#
# NOTE: absolute --out path. sweep_sched runs inference with cwd=/home/ebrahim/Protenix, so a
# relative path silently writes outside tts_abag (a smoke test did exactly that once).
set -o pipefail

REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=$1
LOCKS=$A/phase30_locks
LIST=${LIST:-$REPO/ops/order_phase30.txt}
NEED=${NEED:-250}
STALE_MIN=${STALE_MIN:-180}   # 30 recycles is ~3x slower than the 10-cycle arm; don't reclaim early
mkdir -p "$LOCKS" "$A/sweep_out/phase30" "$A/phase30_logs"
cd "$REPO" || exit 1

n_poses(){ find "$A/sweep_out/phase30/phase30_$1" -name '*_sample_*.cif' 2>/dev/null | wc -l; }

echo "### PHASE30 worker GPU$GPU (MSA_PHASES=10,20 N_cycle=30) $(date +%H:%M)"
while true; do
  claimed=""
  for pdb in $(awk '{print $1}' "$LIST"); do
    [ "$(n_poses "$pdb")" -ge "$NEED" ] && continue
    if [ -d "$LOCKS/$pdb" ] && [ -n "$(find "$LOCKS/$pdb" -maxdepth 0 -mmin +$STALE_MIN 2>/dev/null)" ]; then
      echo "### reclaiming stale lock on $pdb"; rmdir "$LOCKS/$pdb" 2>/dev/null
    fi
    if mkdir "$LOCKS/$pdb" 2>/dev/null; then
      claimed="$pdb"; echo "$GPU" > "$LOCKS/$pdb/gpu"; break
    fi
  done
  [ -z "$claimed" ] && { echo "### GPU$GPU: nothing left to claim, exiting $(date +%H:%M)"; break; }

  echo "### GPU$GPU claimed $claimed ($(n_poses "$claimed")/$NEED) $(date +%H:%M)"
  jf=$A/phase30_one_${claimed}.json
  python3 - "$claimed" "$jf" <<'PYEOF'
import json, sys
p, jf = sys.argv[1], sys.argv[2]
json.dump([{"run": f"phase30_{p}", "pdb": p,
            "json": f"/home/ebrahim/Protenix/foldbench_abag/jsons/{p}-final-updated.json",
            "name": p, "seeds": list(range(101, 151))}], open(jf, "w"))
PYEOF
  PROTENIX_MSA_PHASES=10,20 \
    $PY $REPO/scripts/sweep_sched.py --jobs "$jf" --out "$A/sweep_out/phase30" \
        --logs "$A/phase30_logs" --status "$A/phase30_status_g${GPU}.json" \
        --gpus "$GPU" --n-sample 5 --extra-args '--model.N_cycle 30'
  echo "### GPU$GPU finished $claimed poses=$(n_poses "$claimed") $(date +%H:%M)"
  rm -f "$jf"
done
