#!/usr/bin/env bash
# Score ConforNet complexes as they finish, so results are ready without waiting for the whole sweep.
#
# CHEAP TO RE-RUN: score_confornet.py caches one OST json per pose, so a complex already scored costs
# only a re-parse (seconds). That means this can just rescore everything complete each cycle rather
# than tracking state, which is one less thing to get wrong.
#
# ONLY COMPLETE COMPLEXES: --min-poses 400 skips anything still generating. A complex scored at 250 of
# 400 poses would enter the table as a shallower arm and drag its own averages down, which reads as a
# finding rather than an artefact.
#
# CPU: the scorer pins OST to cores 0-31 and runs 24 jobs, well under the 64-core limit, and touches
# no GPU -- so it can run alongside the sweep.
set -o pipefail

EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/miniconda3/envs/foldbench/bin/python
PFX=${PFX:-v3}
EVERY=${EVERY:-1800}
LOG=$EXP/ops/score_watch_${PFX}.log

cd "$EXP" || exit 1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "score watcher started: prefix=$PFX interval=${EVERY}s"
last_done=-1
while true; do
  done_n=$(find artifacts/confornet -path "*_${PFX}_s*" -name "*.cif" 2>/dev/null \
           | sed "s|.*/confornet/||;s|_${PFX}_s.*||" | sort | uniq -c \
           | awk '$1>=400' | wc -l)

  if [ "$done_n" -gt 0 ] && [ "$done_n" -ne "$last_done" ]; then
    say "$done_n complete complexes -> scoring"
    $PY scripts/score_confornet.py --prefix "$PFX" --jobs 24 >>"$EXP/ops/score_${PFX}.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      say "scorer FAILED rc=$rc -- see ops/score_${PFX}.log"
    else
      rows=$(( $(wc -l < "artifacts/confornet_scores_${PFX}.csv" 2>/dev/null || echo 1) - 1 ))
      say "scored: $rows rows across $done_n complexes"
      # one-line-per-interface summary so the log is readable without loading the csv
      $PY - <<EOF | tee -a "$LOG"
import csv, collections, statistics as st
rows=list(csv.DictReader(open("artifacts/confornet_scores_${PFX}.csv")))
by=collections.defaultdict(list)
for r in rows: by[r["pdb"]+"_"+r["iface"]].append(r)
for k in sorted(by):
    v=by[k]; d=[float(x["dockq"]) for x in v]
    sel=max((float(x["pair_iptm"]) if x["pair_iptm"] else -1, float(x["dockq"])) for x in v)[1]
    tr=[st.mean([float(x["dockq"]) for x in v if int(x["step"])==s]) for s in (5,10,15,20)]
    print(f"    {k:<12} n={len(v):>4} mean={st.mean(d):.3f} oracle={max(d):.3f} "
          f"ipTM-sel={sel:.3f} frac={sum(x>0.23 for x in d)/len(d):.2f} "
          f"step5->20 {tr[0]:.3f}->{tr[3]:.3f}")
EOF
    fi
    last_done=$done_n
  fi
  sleep "$EVERY"
done
