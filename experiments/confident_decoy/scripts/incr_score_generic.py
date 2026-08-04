import json,os,glob,subprocess,time,sys
from concurrent.futures import ThreadPoolExecutor
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
# argv: out_dir  suffix  tag
OUT=sys.argv[1]; SUF=sys.argv[2]; TAG=sys.argv[3]
OST="/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV=dict(os.environ); ENV["LD_LIBRARY_PATH"]="/home/ebrahim/miniconda3/envs/foldbench/lib"
CACHE=f"{SC}/ost_{SUF}_incr"; os.makedirs(CACHE,exist_ok=True)
meta=json.load(open(f"{SC}/meta.json")); GT="/home/ebrahim/FoldBench/examples/ground_truths"
def out(m): print(m,flush=True)
def run_ost(m,n,o):
    if os.path.exists(o) and os.path.getsize(o)>0: return
    subprocess.run(["taskset","-c","0-49",OST,"compare-structures","-m",m,"-r",n,"-o",o,"--fault-tolerant","--min-pep-length","4","--dockq"],env=ENV,capture_output=True,timeout=300)
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
def score_iface(ifc,pdb,c1,c2):
    name=f"{pdb}_{SUF}"; nat=f"{GT}/{pdb}-assembly1.cif"; jobs=[]
    for s in sorted(int(os.path.basename(d).split('_')[1]) for d in glob.glob(f"{OUT}/{name}/seed_*")):
        for k in range(5):
            cif=f"{OUT}/{name}/seed_{s}/predictions/{name}_sample_{k}.cif"; cf=f"{OUT}/{name}/seed_{s}/predictions/{name}_summary_confidence_sample_{k}.json"
            if os.path.exists(cif): jobs.append((cif,cf,f"{CACHE}/{ifc}_{s}_{k}.json"))
    with ThreadPoolExecutor(max_workers=12) as e: list(e.map(lambda j: run_ost(j[0],nat,j[2]),jobs))
    poses=[]
    for cif,cf,o in jobs:
        r=ex(o,cf,c1,c2)
        if r and r[0] is not None: poses.append((r[1],r[0]))
    if not poses: return None
    sel=max(poses,key=lambda x:(x[0] if x[0] is not None else -1))
    return sel[1],sel[0],len(poses)
bypdb={}
for ifc,m in meta.items(): bypdb.setdefault(m["pdb"],[]).append((ifc,m["c1"],m["c2"]))
done=set(); total=len(meta)
out(f"INCR[{TAG}] scorer started: {len(bypdb)} pdbs / {total} interfaces")
t0=time.time()
while len(done)<total and time.time()-t0<40000:
    for pdb,ifs in bypdb.items():
        if all(ifc in done for ifc,_,_ in ifs): continue
        if len(glob.glob(f"{OUT}/{pdb}_{SUF}/seed_*"))<50: continue
        for ifc,c1,c2 in ifs:
            if ifc in done: continue
            r=score_iface(ifc,pdb,c1,c2)
            if r is None: continue
            dq,pi,n=r; done.add(ifc)
            out(f"RESULT[{TAG}] {ifc}: sel_DockQ={dq:.3f} sel_ipTM={pi if pi is not None else 0:.3f} -> {'RESCUED' if dq>=0.23 else 'not-rescued'}  [{len(done)}/{total}]")
    time.sleep(60)
out(f"ALL_DONE[{TAG}]: {len(done)}/{total} interfaces scored")
