"""greedy_leader.py — ipTM-ordered greedy leader clustering (dedup by structure), per Type-1 interface.

Idea (user's): rank seeds by confidence (each seed = its best-pair_iptm diffusion sample), then walk high->low:
first seed opens cluster 1; each next seed joins the nearest existing leader if structurally within T, else
opens a new cluster. Leaders = distinct basins, ordered by ipTM. This DEDUPLICATES the confident-decoy basin
(hundreds of seeds -> one leader), so the good basin's leader should surface at a low rank -- unlike pose/seed
top-K, which is clogged with redundant decoy copies.

Reports, per interface and per threshold T: rank of the first good-dock leader, and #leaders.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from sklearn.preprocessing import StandardScaler

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
T1 = [('8a44','A','C'),('8a44','A','D'),('7yxu','A','C'),('8vui','B','C'),('8w84','B','C'),
      ('8aok','A','B'),('8oni','B','E'),('8ts0','A','C'),('8ts0','B','C'),('7zoz','A','C')]
AGAB = {}
for r in csv.DictReader(open("results/csv_files/decoy_structural_analysis.csv")):
    if r.get('pred_ag_chain') and r.get('pred_ab_chain'):
        AGAB[(r['cx'], r['c1'], r['c2'])] = (r['pred_ag_chain'], r['pred_ab_chain'])
# per interface: seed -> best-ipTM sample (iptm, dockq, sample_k)
BEST = defaultdict(lambda: defaultdict(lambda: (-1.0, 0.0, 0)))
T1SET={tuple(t) for t in T1}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key=(r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in T1SET: continue
    try:
        pv=float(r['pair_iptm']); dq=float(r['dockq']); s=int(r['seed']); k=int(r['sample'])
        if pv>BEST[key][s][0]: BEST[key][s]=(pv,dq,k)
    except: pass

def fast_reps(cif):
    with open(cif) as f: lines=f.readlines()
    cols={}; i=0
    for i,ln in enumerate(lines):
        if ln.startswith('_atom_site.'): cols[ln.strip()]=len(cols)
        elif ln.startswith('ATOM'): break
    C=lambda n: cols['_atom_site.'+n]
    ich,iseq,inm=C('auth_asym_id'),C('label_seq_id'),C('label_atom_id')
    ix,iy,iz=C('Cartn_x'),C('Cartn_y'),C('Cartn_z')
    res={}; order=[]
    for ln in lines[i:]:
        if not ln.startswith('ATOM'): continue
        p=ln.split(); k=(p[ich],p[iseq])
        if k not in res: res[k]={}; order.append(k)
        res[k][p[inm]]=(float(p[ix]),float(p[iy]),float(p[iz]))
    out=defaultdict(list)
    for k in order:
        a=res[k]; xyz=a.get('CB',a.get('CA'))
        if xyz: out[k[0]].append(np.array(xyz,dtype=np.float32))
    return {c:np.stack(v) for c,v in out.items()}

def fw(args):
    s,k,cx,ag,ab,agi,abi=args
    cif=DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
    if not cif.exists(): return (s,None)
    try:
        R=fast_reps(cif)
        if ag not in R or ab not in R: return (s,None)
        A=R[ag][agi]; B=R[ab][abi]; iu=np.triu_indices(len(A),1)
        intra=np.linalg.norm(A[:,None,:]-A[None,:,:],axis=-1)[iu]
        cross=np.linalg.norm(A[:,None,:]-B[None,:,:],axis=-1).ravel()
        return (s,np.concatenate([intra,cross]).astype(np.float32))
    except Exception: return (s,None)

def leader_rank(F, order, dq, T):
    """greedy leader clustering in ipTM order; return (good_leader_rank, n_leaders)."""
    leaders=[]  # feature-row indices, in creation (ipTM) order
    for i in order:
        if leaders:
            d=np.linalg.norm(F[leaders]-F[i], axis=1)
            if d.min()<T: continue          # joins an existing leader
        leaders.append(i)                    # new leader
    for rank,li in enumerate(leaders,1):
        if dq[li]>0.4: return rank, len(leaders)
    return None, len(leaders)

def process(key):
    cx,c1,c2=key
    if key not in AGAB: return (key,None)
    ag,ab=AGAB[key]
    seeds=sorted(BEST[key].keys())
    reps=[(s,BEST[key][s][2]) for s in seeds]          # (seed, best sample k)
    ref=None
    for s,k in reps:
        f=DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
        if f.exists(): ref=fast_reps(f); break
    if not ref or ag not in ref or ab not in ref: return (key,None)
    nag,nab=len(ref[ag]),len(ref[ab])
    agi=np.linspace(0,nag-1,min(60,nag)).astype(int); abi=np.linspace(0,nab-1,min(60,nab)).astype(int)
    with Pool(30) as p:
        res=dict(p.map(fw,[(s,k,cx,ag,ab,agi,abi) for s,k in reps]))
    rows=[(s,res[s]) for s in seeds if res.get(s) is not None]
    if len(rows)<30: return (key,None)
    F=StandardScaler().fit_transform(np.stack([f for _,f in rows]))
    iptm=np.array([BEST[key][s][0] for s,_ in rows]); dq=np.array([BEST[key][s][1] for s,_ in rows])
    order=list(np.argsort(-iptm))                       # high ipTM -> low
    # threshold sweep, calibrated on pairwise distances (sample 4000 pairs)
    n=len(F); ri=np.random.RandomState(0).randint(0,n,(4000,2))
    pd=np.linalg.norm(F[ri[:,0]]-F[ri[:,1]],axis=1); pd=pd[pd>0]
    Ts={p:np.percentile(pd,p) for p in [2,5,10,20,35]}
    out={}
    for p,T in Ts.items():
        gr,nl=leader_rank(F,order,dq,T); out[p]=(gr,nl)
    return (key, out)

if __name__=="__main__":
    print(f"{'iface':>12} | " + "  ".join(f"p{p}:(rank,#lead)" for p in [2,5,10,20,35]))
    for t in T1:
        key,out=process(tuple(t))
        if out is None: print(f"{t[0]+' '+t[1]+'-'+t[2]:>12} | (skip)"); continue
        cells=[f"{str(out[p][0]):>3}/{out[p][1]:<4}" for p in [2,5,10,20,35]]
        print(f"{t[0]+' '+t[1]+'-'+t[2]:>12} | " + "  ".join(cells), flush=True)
    print("cells = goodLeaderRank / numLeaders, at pairwise-distance percentile thresholds p2..p35")
    print("DONE", flush=True)
