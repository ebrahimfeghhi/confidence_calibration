#!/usr/bin/env bash
# Score PHASE30 (3-phase MSA curriculum, 30 recycles) complexes as they finish, so results are ready without waiting for the whole sweep.
#
# CHEAP TO RE-RUN: score_msaoff5.py caches one OST json per pose, so an already-scored complex costs
# only a re-parse. That means we can rescore everything complete each cycle instead of tracking state.
#
# ONLY COMPLETE COMPLEXES (250 poses). A partial complex would enter the averages as a shallower arm.
#
# REPORTS UNSCOREABLE COMPLEXES EXPLICITLY. 8f60 finished all 250 poses and produced ZERO rows,
# because the antibody never contacts the antigen in any pose -- OST finds only the A-B interface, so
# DockQ for the labelled A-C/B-C pairs is undefined and the scorer correctly refuses to guess. That
# silence looks identical to "not finished yet", so it is called out rather than left to be noticed.
set -o pipefail

EXP=/home/ebrahim/tts_abag/confident_decoy_experiments
PY=/home/ebrahim/miniconda3/envs/foldbench/bin/python
EVERY=${EVERY:-600}
LOG=$EXP/ops/score_watch_phase30.log

cd "$EXP" || exit 1
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "score watcher started (interval ${EVERY}s)"
last=-1
while true; do
  done_n=$(find artifacts/sweep_out/phase30 -name '*_sample_*.cif' 2>/dev/null \
           | sed 's|.*/phase30/phase30_||;s|/.*||' | sort | uniq -c | awk '$1>=250' | wc -l)
  if [ "$done_n" -gt 0 ] && [ "$done_n" -ne "$last" ]; then
    say "$done_n complete complexes -> scoring"
    $PY scripts/score_msaoff5.py --arm phase30 --jobs 24 >>"$EXP/ops/score_phase30.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      say "scorer FAILED rc=$rc -- see ops/score_phase30.log"
    else
      scored=$(awk -F, 'NR>1{print $1}' artifacts/phase30_scores.csv | sort -u | wc -l)
      say "scored $scored of $done_n complete complexes"
      # any complex that finished but produced no rows -- do not let it vanish silently
      comp=$(find artifacts/sweep_out/phase30 -name '*_sample_*.cif' | sed 's|.*/phase30/phase30_||;s|/.*||' \
             | sort | uniq -c | awk '$1>=250{print $2}' | sort)
      got=$(awk -F, 'NR>1{print $1}' artifacts/phase30_scores.csv | sort -u)
      miss=$(comm -23 <(echo "$comp") <(echo "$got") | tr '\n' ' ')
      [ -n "$miss" ] && say "  UNSCOREABLE (finished but no labelled interface found): $miss"
      $PY - <<'EOF' | tee -a "$LOG"
import csv, json, glob, re, collections, statistics as st
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
ifm=collections.defaultdict(list)
for r in csv.DictReader(open("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")):
    ifm[r["pdb_id"].split("-")[0]].append((r["antibody_chain"],r["antigen_chain"]))
mo=collections.defaultdict(list)
for r in csv.DictReader(open("artifacts/phase30_scores.csv")):
    mo[r['pdb']+'_'+r['iface']].append((float(r['pair_iptm']) if r['pair_iptm'] else -1, float(r['dockq'])))
free=collections.defaultdict(list)
for f in glob.glob("artifacts/abag_all_ost/*.json"):
    m=re.match(r"([0-9a-z]{4})_(\d+)_(\d+)\.json$", f.split("/")[-1])
    if not m: continue
    pdb,s,k=m.groups()
    try:
        j=json.load(open(f)); M=json.load(open(f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
    except Exception: continue
    qs,ifs=j.get("dockq") or [],j.get("dockq_interfaces") or []
    for c1,c2 in ifm.get(pdb,[]):
        for i,t in enumerate(ifs):
            if i<len(qs) and {t[0].split("-")[0],t[1].split("-")[0]}=={c1,c2}:
                a,b=ord(t[2])-65,ord(t[3])-65
                free[f"{pdb}_{c1}{c2}"].append((M[a][b] if (0<=a<len(M) and 0<=b<len(M)) else -1, qs[i])); break
keys=sorted(set(mo)&set(free))
if keys:
    F=[max(free[k])[1] for k in keys]; M_=[max(mo[k])[1] for k in keys]
    w=sum(1 for a,b in zip(M_,F) if a-b>0.01); t=sum(1 for a,b in zip(M_,F) if abs(a-b)<=0.01)
    print(f"    n={len(keys)} interfaces | selected DockQ  free {st.mean(F):.3f} -> phase30 {st.mean(M_):.3f} "
          f"({st.mean(M_)-st.mean(F):+.3f}) | median {st.median(F):.3f} -> {st.median(M_):.3f} "
          f"| better {w} tied {t} worse {len(keys)-w-t}")
EOF
    fi
    last=$done_n
  fi
  sleep "$EVERY"
done
