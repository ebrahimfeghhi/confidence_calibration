import json,os,glob,subprocess,collections,sys
from concurrent.futures import ThreadPoolExecutor
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad"
OST="/home/ebrahim/miniconda3/envs/foldbench/bin/ost"; ENV=dict(os.environ); ENV["LD_LIBRARY_PATH"]="/home/ebrahim/miniconda3/envs/foldbench/lib"
OUT=f"{SC}/selftmpl/abonly_out"; os.makedirs(f"{SC}/selftmpl/ost_abonly",exist_ok=True)
which=sys.argv[1] if len(sys.argv)>1 else "both"   # decoys|goods|both
def run_ost(m,n,o):
    if os.path.exists(o) and os.path.getsize(o)>0: return
    subprocess.run(["taskset","-c","0-39",OST,"compare-structures","-m",m,"-r",n,"-o",o,"--fault-tolerant","--min-pep-length","4","--dockq"],env=ENV,capture_output=True,timeout=300)
def ex(o,cf,c1,c2):
    try: d=json.load(open(o))
    except: return None
    dqv=None;mc=None
    for k,t in enumerate(d.get("dockq_interfaces") or []):
        if {t[0],t[1]}=={c1,c2}: q=d.get("dockq") or []; dqv=q[k] if k<len(q) else None; mc=(t[2],t[3]); break
    pi=None
    try:
        c=json.load(open(cf)); M=c.get("chain_pair_iptm")
        if mc and M:
            i=ord(mc[0])-ord('A');j=ord(mc[1])-ord('A')
            if 0<=i<len(M) and 0<=j<len(M): pi=M[i][j]
    except: pass
    return dqv,pi
ifaces=[]
if which in ("decoys","both"):
    for ifc,m in json.load(open(f"{SC}/selftmpl/meta.json")).items(): ifaces.append(("decoy",ifc,m["pdb"],m["c1"],m["c2"]))
if which in ("goods","both"):
    for x in json.load(open(f"{SC}/selftmpl/safety15.json")): ifaces.append(("good",x["iface"],x["pdb"],x["c1"],x["c2"]))
jobs=[]
for grp,ifc,pdb,c1,c2 in ifaces:
    nat=f"/home/ebrahim/FoldBench/examples/ground_truths/{pdb}-assembly1.cif"; name=f"{pdb}_abonly"
    for s in sorted(int(os.path.basename(d).split('_')[1]) for d in glob.glob(f"{OUT}/{name}/seed_*")):
        for k in range(5):
            cif=f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"; cf=f"{OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if os.path.exists(cif): jobs.append((grp,ifc,cif,cf,nat,f"{SC}/selftmpl/ost_abonly/{ifc}_{s}_{k}.json",c1,c2))
print("OST on",len(jobs),flush=True)
with ThreadPoolExecutor(max_workers=16) as e: list(e.map(lambda j: run_ost(j[2],j[4],j[5]),jobs))
res=collections.defaultdict(list); grpof={}
for grp,ifc,cif,cf,nat,o,c1,c2 in jobs:
    grpof[ifc]=grp
    r=ex(o,cf,c1,c2)
    if r and r[0] is not None: res[ifc].append({"pair_iptm":r[1],"dockq":r[0]})
json.dump(res,open(f"{SC}/selftmpl/abonly_scores.json","w"))
# summary: selection = top pair_iptm pose's dockq
def sel(v): return max(v,key=lambda x:(x["pair_iptm"] if x["pair_iptm"] is not None else -1))
for grp in ["decoy","good"]:
    ig=[i for i in res if grpof.get(i)==grp]
    if not ig: continue
    hit=[i for i in ig if sel(res[i])["dockq"]>=0.23]
    label="rescued" if grp=="decoy" else "preserved"
    print(f"{grp}: {label} {len(hit)}/{len(ig)} -> {sorted(hit)}",flush=True)
print("SCORE_ABONLY_DONE",flush=True)
