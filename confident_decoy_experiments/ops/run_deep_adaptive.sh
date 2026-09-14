#!/usr/bin/env bash
# ADAPTIVE DEEP SCALING for the interfaces that still have no acceptable pose.
#
# Per complex: generate 50 more seeds -> score them -> check. Stop as soon as EITHER
#   (a) the ipTM-SELECTED pose reaches DockQ >= 0.23 for every labelled interface, or
#   (b) the complex reaches 1024 free-run seeds.
# The early stop is on SELECTION, not oracle: a complex whose selector already works needs no more
# seeds, whereas one with a good pose the selector keeps missing is a selection problem that more
# sampling cannot fix.
#
# WHY ADAPTIVE: a flat "run everything to 1024" costs ~7,700 seeds across these 9 complexes. Most of
# that is wasted if a complex succeeds at seed 300 -- and 8 of 40 interfaces already flipped status
# purely from scoring poses that were sitting unscored on disk, so the marginal value of each batch
# has to be re-measured rather than assumed.
#
# WORK-STEALING via atomic mkdir claims, same as run_msaoff5.sh / run_phase30.sh: GPUs run
# non-overlapping complexes and whoever frees up first takes the next unclaimed one.
#
# SEED NUMBERING continues past whatever already exists (max existing + 1), so new batches never
# collide with the poses already on disk.
set -o pipefail

REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/Protenix/.venv/bin/python
A=$REPO/artifacts
GPU=$1
LOCKS=$A/deep_adaptive_locks
LIST=${LIST:-$A/still_failing.txt}
BATCH=${BATCH:-50}
MAXSEED=${MAXSEED:-1024}
THRESH=0.23
mkdir -p "$LOCKS" "$A/sweep_out/deepadapt" "$A/deep_adaptive_logs"
cd "$REPO" || exit 1

# free-run seeds already on disk, across every root that holds them
n_seeds(){
  $PY - "$1" <<'PYEOF'
import glob, sys, re
p = sys.argv[1]
P = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
A = "/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/sweep_out"
s = set()
for pat in (f"{P}/{p}/seed_*", f"{A}/freedeep{p}/**/seed_*", f"{A}/deepadapt/deepadapt_{p}/**/seed_*"):
    for x in glob.glob(pat, recursive=True):
        m = re.search(r"seed_(\d+)$", x)
        if m:
            s.add(int(m.group(1)))
print(len(s), (max(s) if s else 0))
PYEOF
}

# every labelled interface of this complex has a SELECTED pose at or above threshold?
sel_ok(){
  $PY - "$1" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
f = f"/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts/deep_adaptive_{p}.json"
if not os.path.exists(f):
    print("0"); raise SystemExit
d = json.load(open(f))
rows = [v for k, v in d.items() if k.startswith(p) and v.get("status") != "not_scored"]
print("1" if rows and all(v.get("sel_dockq", 0) >= 0.23 for v in rows) else "0")
PYEOF
}

echo "### DEEP-ADAPTIVE worker GPU$GPU (batch=$BATCH, cap=$MAXSEED) $(date +%H:%M)"
while true; do
  claimed=""
  for pdb in $(awk '{print $1}' "$LIST"); do
    read -r n mx <<<"$(n_seeds "$pdb")"
    [ "${n:-0}" -ge "$MAXSEED" ] && continue
    [ "$(sel_ok "$pdb")" = "1" ] && continue
    if mkdir "$LOCKS/$pdb" 2>/dev/null; then claimed="$pdb"; echo "$GPU" > "$LOCKS/$pdb/gpu"; break; fi
  done
  [ -z "$claimed" ] && { echo "### GPU$GPU: nothing left to claim, exiting $(date +%H:%M)"; break; }

  while true; do
    read -r n mx <<<"$(n_seeds "$claimed")"
    if [ "$n" -ge "$MAXSEED" ]; then echo "### $claimed reached $n seeds (cap) -- done"; break; fi
    if [ "$(sel_ok "$claimed")" = "1" ]; then echo "### $claimed SELECTION SUCCEEDED at $n seeds -- early stop"; break; fi
    lo=$((mx+1)); hi=$((mx+BATCH))
    echo "### GPU$GPU $claimed: $n seeds -> generating $lo-$hi $(date +%H:%M)"
    jf=$A/deepadapt_${claimed}.json
    $PY - "$claimed" "$jf" "$lo" "$hi" <<'PYEOF'
import json, sys
p, jf, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
json.dump([{"run": f"deepadapt_{p}", "pdb": p,
            "json": f"/home/ebrahim/Protenix/foldbench_abag/jsons/{p}-final-updated.json",
            "name": p, "seeds": list(range(lo, hi + 1))}], open(jf, "w"))
PYEOF
    $PY $REPO/scripts/sweep_sched.py --jobs "$jf" --out "$A/sweep_out/deepadapt" \
        --logs "$A/deep_adaptive_logs" --status "$A/deepadapt_status_g${GPU}.json" \
        --gpus "$GPU" --n-sample 5 || { echo "### $claimed generation failed"; break; }
    # score the whole complex (cached, so only the new poses cost anything)
    ABAG_NWORK=6 ABAG_TASKSET=0-23 $PY $REPO/scripts/abag_all_101_150.py \
        --complexes "$claimed" --all-seeds --out "$A/deep_adaptive_${claimed}.json" \
        >> "$A/deep_adaptive_logs/score_${claimed}.log" 2>&1
  done
  rm -f "$A/deepadapt_${claimed}.json"
done
