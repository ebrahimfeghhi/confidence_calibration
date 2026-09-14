import json,os,glob,subprocess
from concurrent.futures import ThreadPoolExecutor
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
GT="/home/ebrahim/FoldBench/examples/ground_truths"
OST="/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV=dict(os.environ); ENV["LD_LIBRARY_PATH"]="/home/ebrahim/miniconda3/envs/foldbench/lib"
CACHE=f"{SC}/ost_freeDQ"; os.makedirs(CACHE,exist_ok=True)
meta=json.load(open(f"{SC}/meta.json"))
ifpp={}
for ifc,m in meta.items(): ifpp.setdefault(m["pdb"],[]).append((ifc,m["c1"],m["c2"]))
def run_ost(m,n,o):
    # delegates to eval/ost_dockq.py -- the single OST entry point. This used to be a
    # local copy; the copies drifted and five of them still passed --rigid-scores, which
    # crashes on small interfaces and loses the whole score.
    return _ost_run(m, n, o)

def dq_of(o,c1,c2):
    try: d=json.load(open(o))
    except: return None
    for k,t in enumerate(d.get("dockq_interfaces") or []):
        if {t[0],t[1]}=={c1,c2}: q=d.get("dockq") or []; return q[k] if k<len(q) else None
    return None
# build OST jobs for all poses of all decoy pdbs (seeds 101-150)
jobs=[]
for pdb in ifpp:
    nat=f"{GT}/{pdb}-assembly1.cif"
    for s in range(101,151):
        for k in range(5):
            cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
            if os.path.exists(cif): jobs.append((pdb,s,k,cif,nat,f"{CACHE}/{pdb}_{s}_{k}.json"))
print("free DockQ OST jobs:",len(jobs),flush=True)
with ThreadPoolExecutor(max_workers=16) as e: list(e.map(lambda j: run_ost(j[3],j[4],j[5]),jobs))
picks={}
best={}   # pdb -> (dockq, seed, sample, cif)
for pdb,s,k,cif,nat,o in jobs:
    mx=-1
    for ifc,c1,c2 in ifpp[pdb]:
        dq=dq_of(o,c1,c2)
        if dq is not None: mx=max(mx,dq)
    if mx>=0 and (pdb not in best or mx>best[pdb][0]):
        best[pdb]=(mx,s,k,cif)
for pdb,(dq,s,k,cif) in best.items(): picks[pdb]={"seed":s,"sample":k,"dockq":dq,"cif":cif}
json.dump(picks,open(f"{SC}/topdockq_picks.json","w"),indent=1)
print("TOPDOCKQ_PICKS_DONE")
for pdb,p in picks.items(): print(f"  {pdb}: seed{p['seed']}/s{p['sample']} bestDockQ={p['dockq']:.3f}")
