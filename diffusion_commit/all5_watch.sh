#!/usr/bin/env bash
# Every 5 min: sweep progress, the donor-vs-all5 comparison so far, and run health. REMEDIATES the one
# thing it can: a GPU that has lost its holder and sits empty, which is how a neighbour takes our card.
#
# A watcher that only prints is not enough -- this project has a case where the health log correctly
# flagged a wedged worker for 16 hours while it burned a GPU and nobody was reading.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
DC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=/home/ebrahim/Protenix/.venv/bin/python
CSV=$REPO/artifacts/commit11/results.csv
LOG=${LOG:-$REPO/artifacts/all5_watch.log}
STOP=${STOP:-$REPO/artifacts/ALL5_WATCH_STOP}
GPUS=${GPUS:-"5 6"}
PERIOD=${PERIOD:-300}
ME=$(id -un)
rm -f "$STOP"
echo "[all5_watch] every ${PERIOD}s -> $LOG ; stop: touch $STOP"

while true; do
  [ -f "$STOP" ] && { echo "[all5_watch] stopping" >> "$LOG"; exit 0; }
  {
    echo "===== $(date '+%m-%d %H:%M:%S') ====="
    tail -2 "$REPO/artifacts/all5sweep.log" 2>/dev/null | sed 's/^/  sweep: /'

    # --- the paired comparison: recycle 10 (donor) vs the 11th recycle with all 5 poses (all5_k1)
    $PY - "$CSV" <<'EOF'
import csv, sys, collections, statistics as st
AP=0.23
# THE END GOAL: per interface, pool every pose of every seed and take the single highest-ipTM one --
# recycle 10 vs the 11th recycle with all 5 poses. Restricted to seeds where BOTH arms exist, because an
# unmatched pool would differ in depth as well as in method and the comparison would be meaningless.
seen=set(); by=collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(sys.argv[1])):
    if r["arm"] not in ("donor","all5_k1") or not r["iptm"]: continue
    k=(r["iface"],r["seed"],r["arm"],r["sample"],r["dockq"])
    if k in seen: continue
    seen.add(k)
    by[r["iface"]][(r["seed"],r["arm"])].append((float(r["dockq"]),float(r["iptm"])))
rows=[]
for iface,d in by.items():
    seeds=[s for s in {k[0] for k in d}
           if len(d.get((s,"donor"),[]))>=5 and len(d.get((s,"all5_k1"),[]))>=5]
    if not seeds: continue
    P10=[x for s in seeds for x in d[(s,"donor")]]
    P11=[x for s in seeds for x in d[(s,"all5_k1")]]
    s10=max(P10,key=lambda x:x[1]); s11=max(P11,key=lambda x:x[1])
    rows.append((iface,len(seeds),s10[0],s11[0],max(x[0] for x in P10),max(x[0] for x in P11)))
print(f"  interfaces with matched pools: {len(rows)}   (seeds: {min((r[1] for r in rows), default=0)}-{max((r[1] for r in rows), default=0)})")
if rows:
    p10=[r[2] for r in rows]; p11=[r[3] for r in rows]
    o10=[r[4] for r in rows]; o11=[r[5] for r in rows]
    w=sum(1 for r in rows if r[3]>r[2]+1e-9); l=sum(1 for r in rows if r[3]<r[2]-1e-9)
    print(f"  POOLED ipTM pick   recycle10 {st.mean(p10):.3f}  ->  recycle11+all5 {st.mean(p11):.3f}   "
          f"delta {st.mean(p11)-st.mean(p10):+.3f}")
    print(f"  acceptable picks   recycle10 {sum(1 for x in p10 if x>=AP)}/{len(rows)}  ->  "
          f"{sum(1 for x in p11 if x>=AP)}/{len(rows)}      better/worse/tied {w}/{l}/{len(rows)-w-l}")
    print(f"  oracle in pool     recycle10 {st.mean(o10):.3f}  ->  {st.mean(o11):.3f}")
    for iface,n,a10,a11,_,_ in sorted(rows, key=lambda r:-(r[3]-r[2]))[:4]:
        print(f"    {iface:<12} n={n:>2}  {a10:.3f} -> {a11:.3f}  ({a11-a10:+.3f})")
EOF

    # --- health, and re-hold any empty card of ours
    for g in $GPUS; do
      ours=0; other=""
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
        u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
        [ -z "$u" ] && continue
        [ "$u" = "$ME" ] && ours=1 || other="$other $u"
      done
      h=$(tmux has-session -t "hhold$g" 2>/dev/null && echo held || echo UNHELD)
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$g" 2>/dev/null)
      echo "  GPU$g holder=$h util=$util ours=$ours others=${other:-none}"
      if [ "$h" = "UNHELD" ] && [ -z "$other" ]; then
        rm -f "$REPO/artifacts/.hold_stop_gpu$g"
        tmux new-session -d -s "hhold$g" -c "$REPO" \
          "CUDA_VISIBLE_DEVICES=$g $PY $REPO/scripts/gpu_hold.py --gib 10 --stop $REPO/artifacts/.hold_stop_gpu$g"
        echo "    -> re-held GPU$g (was empty and unheld)"
      fi
    done
    tmux has-session -t sweep 2>/dev/null || echo "  !! SWEEP SESSION GONE"
    for g in $GPUS; do
      tail -60 "$REPO/artifacts/all5_g$g.log" 2>/dev/null | grep -cE "!!|Traceback" | \
        awk -v g=$g '$1>0 {print "  !! GPU"g": "$1" error lines in the last 60 log lines"}'
    done
    df -h /home 2>/dev/null | tail -1 | awk '{print "  disk: "$3" used, "$4" free"}'
    echo
  } >> "$LOG" 2>&1
  sleep "$PERIOD"
done
