#!/usr/bin/env bash
# MSA OFF for recycles 0-4, ON for 5-9. Everything else stock (subsampling ON, dropout ON).
# PROTENIX_MSA_FROM_CYCLE=5 skips the MSA module for the first five cycles, so the trunk builds z from
# templates + sequence alone before any coevolutionary signal arrives.
# Verified not to be a no-op: same seed, k=0 vs k=5 gives 11.98 A mean coordinate deviation on 8qf4.
#
# WORK-STEALING VIA ATOMIC CLAIMS. sweep_sched.py's lock is a threading.Lock -- it coordinates threads
# inside ONE process and nothing across processes. Running three schedulers on one jobs file therefore
# had all three build the SAME shortest-first queue and run the SAME complex at once, writing to the
# same output dir (observed: all 3 GPUs on 8trs, two with identical seed lists 101-150).
#
# Static per-GPU slices fix the overlap but waste capacity -- a GPU that finishes its slice idles while
# another still has work. Instead each GPU claims ONE complex at a time by `mkdir` on a lock dir, which
# is atomic on POSIX: exactly one process can create a given directory, so exactly one GPU runs a given
# complex. When a GPU finishes it loops back and claims whatever is still unclaimed, so load balances
# itself and no slice can be left stranded.
#
# A stale lock (worker killed mid-complex) blocks a retry, so locks older than STALE_MIN with an
# incomplete output dir are reclaimed.
set -o pipefail

REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=$1
LOCKS=$A/msaoff5_locks
# Full 49-complex train set, ascending token count so the cheap complexes bank results first.
# The 11 beyond the old 38-list (1192-2361 tokens, up to 8xnh) were excluded from the ConforNet sweep
# only because TRAINING needs backprop memory; this arm is forward-only inference and every one of the
# 11 has already folded 50+ seeds at --n-sample 5 in the free arm, so all are proven to fit.
# Not sweep_38_order.txt -- chain_confornet.sh and sweep_selftmpl.sh still read that file.
LIST=${LIST:-$REPO/ops/sweep_49_order.txt}
NEED=${NEED:-250}          # poses per complex = 50 seeds x 5 samples
STALE_MIN=${STALE_MIN:-90} # reclaim a lock older than this if its complex is still incomplete
mkdir -p "$LOCKS"

n_poses(){ find "$A/sweep_out/msaoff5/msaoff5_$1" -name '*_sample_*.cif' 2>/dev/null | wc -l; }

# Complexes owned by a STILL-RUNNING static-slice worker (the pre-fix launch). Those workers do not
# take locks, so claiming one of their complexes would recreate the very overlap this replaces. Read
# the job files off the live schedulers' command lines, so ownership is released automatically as each
# old worker exits and this narrows to nothing once the last one is gone.
static_owned(){
  for jf in $(pgrep -af "sweep_sched.py" | grep -oP -- "--jobs \K\S+msaoff5_jobs_g\d+\.json" | sort -u); do
    [ -f "$jf" ] && python3 -c "import json,sys;print('\n'.join(j['pdb'] for j in json.load(open(sys.argv[1]))))" "$jf"
  done | sort -u
}

echo "### MSA_FROM_CYCLE=5 worker on GPU$GPU $(date +%H:%M)"
while true; do
  claimed=""
  OWNED=$(static_owned)
  for pdb in $(awk '{print $1}' "$LIST"); do
    [ "$(n_poses "$pdb")" -ge "$NEED" ] && continue
    grep -qx "$pdb" <<<"$OWNED" && continue   # a live static worker still owns it
    # reclaim a stale lock whose complex never finished
    if [ -d "$LOCKS/$pdb" ] && [ -n "$(find "$LOCKS/$pdb" -maxdepth 0 -mmin +$STALE_MIN 2>/dev/null)" ]; then
      echo "### reclaiming stale lock on $pdb"; rmdir "$LOCKS/$pdb" 2>/dev/null
    fi
    if mkdir "$LOCKS/$pdb" 2>/dev/null; then     # atomic: only one GPU wins
      claimed="$pdb"; echo "$GPU" > "$LOCKS/$pdb/gpu"; break
    fi
  done
  [ -z "$claimed" ] && { echo "### GPU$GPU: nothing left to claim, exiting $(date +%H:%M)"; break; }

  echo "### GPU$GPU claimed $claimed ($(n_poses "$claimed")/$NEED) $(date +%H:%M)"
  python3 - "$claimed" <<'PYEOF'
import json, sys
JD="/home/ebrahim/Protenix/foldbench_abag/jsons"
p=sys.argv[1]
json.dump([{"run":f"msaoff5_{p}","pdb":p,"json":f"{JD}/{p}-final-updated.json",
            "name":p,"seeds":list(range(101,151))}],
          open(f"/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/msaoff5_one_{p}.json","w"))
PYEOF
  PROTENIX_MSA_FROM_CYCLE=5 \
    $PY $REPO/scripts/sweep_sched.py --jobs $A/msaoff5_one_${claimed}.json \
        --out $A/sweep_out/msaoff5 --logs $A/msaoff5_logs \
        --status $A/msaoff5_status_g${GPU}.json --gpus $GPU --n-sample 5
  echo "### GPU$GPU finished $claimed poses=$(n_poses "$claimed") $(date +%H:%M)"
  # leave the lock in place on success so nobody re-runs it; the pose-count check also guards this
done
