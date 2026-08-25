#!/bin/bash
# Phase 2: once the weight-noise ladder is done, fill in the both-OFF (no subsampling, no dropout)
# 10-seed arm for any train complex that does not already have it.
#
# The existing sigma=0 data is keyed per INTERFACE (dose_s0_<iface>) while inference runs per
# COMPLEX, so "already has 10 seeds" is judged per complex: if every interface of that pdb already
# has a complete 10-seed dose_s0 arm, the complex is skipped rather than re-folded.
set -uo pipefail
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments; A=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$REPO/ops/wnm_chain.log"; }

say "waiting for the weight-noise ladder to finish"
while true; do
    left=0
    while read -r r; do
        [ -z "$r" ] && continue
        w=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$A/${r}_jobs.json')))*5)")
        h=$(find $A/sweep_out/$r -name '*_sample_*.cif' 2>/dev/null | wc -l)
        [ "$h" -lt "$w" ] && left=$((left+1))
    done < "$A/wnm_queue.txt"
    [ "$left" -eq 0 ] && break
    say "ladder still running ($left arms incomplete)"
    sleep 900
done
say "ladder complete -- building the both-off gap-fill arms"

$PY - <<'PYEOF' | tee -a "$REPO/ops/wnm_chain.log"
import json, os, glob, collections
A="/home/ebrahim/tts_abag/confident_decoy_experiments/artifacts"
JD="/home/ebrahim/Protenix/foldbench_abag/jsons"
T=json.load(open(f"{A}/train_complexes.json"))
ALL=json.load(open(f"{A}/abag_all_101_150.json"))
man=json.load(open(f"{A}/wnoise_manifest.json"))
SEEDS=list(range(101,111))
by=collections.defaultdict(list)
for i,v in ALL.items(): by[v["pdb"]].append(i)

def complete(arm):
    jf=f"{A}/{arm}_jobs.json"
    if not os.path.exists(jf): return False
    want=sum(len(j["seeds"]) for j in json.load(open(jf)))*5
    return len(glob.glob(f"{A}/sweep_out/{arm}/**/*_sample_*.cif", recursive=True)) >= want

queue=[]
for pdb in T["complexes"]:
    if all(complete(f"dose_s0_{i}") for i in by[pdb]):
        print(f"  skip {pdb}: already has 10 seeds both-off"); continue
    arm=f"bo_{pdb}"
    json.dump([{"run":arm,"pdb":pdb,"json":f"{JD}/{pdb}-final-updated.json",
                "name":pdb,"seeds":SEEDS}], open(f"{A}/{arm}_jobs.json","w"), indent=1)
    man[arm]={"sigma":0.0,"scope":"none","pdb":pdb,
              "dropout":"off","subsampling":"off","percycle":False}
    queue.append(arm)
json.dump(man, open(f"{A}/wnoise_manifest.json","w"), indent=1)
open(f"{A}/bo_queue.txt","w").write("\n".join(queue)+("\n" if queue else ""))
print(f"  built {len(queue)} both-off arms ({len(queue)*10} seeds)")
PYEOF

n=$(wc -l < "$A/bo_queue.txt" 2>/dev/null || echo 0)
if [ "$n" -eq 0 ]; then say "no gap-fill needed -- done"; exit 0; fi
say "starting both-off workers on the gpus the ladder was using"
for g in 1 5 6 7; do
    tmux has-session -t bow$g 2>/dev/null && continue
    tmux new-session -d -s bow$g -c $REPO
    tmux send-keys -t bow$g "bash $REPO/ops/bo_worker.sh $g 2>&1 | tee -a $REPO/ops/bow$g.log" Enter
done
say "both-off phase launched ($n arms)"
