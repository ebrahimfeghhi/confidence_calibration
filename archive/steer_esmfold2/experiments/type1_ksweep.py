"""type1_ksweep.py — Q3: for each Type-1 interface, how large must K (number of structural clusters) be
before a *representative (medoid)* of some cluster is a good dock? This decouples from ipTM: it asks how
finely you must slice the model's structural hypotheses before the good basin appears as its own cluster.

Full pool (all 5120 poses/interface). Fast custom cif parser (rep = CB, CA fallback). Feature = intra-antigen
+ antigen x antibody cross distogram (subsampled 60 res/chain). Build Ward linkage once, cut at each K,
take each cluster's medoid (pose closest to cluster centroid in PCA space), and record the min K at which a
medoid has DockQ>0.4.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

DEEP = Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0       # PCA seed (determinism / variance check)
T1 = [('8a44','A','C'),('8a44','A','D'),('7yxu','A','C'),('8vui','B','C'),('8w84','B','C'),
      ('8aok','A','B'),('8oni','B','E'),('8ts0','A','C'),('8ts0','B','C'),('7zoz','A','C')]
KS = [2,3,5,10,15,20,30,50,75,100,150,200]

AGAB = {}
for r in csv.DictReader(open("results/csv_files/decoy_structural_analysis.csv")):
    if r.get('pred_ag_chain') and r.get('pred_ab_chain'):
        AGAB[(r['cx'], r['c1'], r['c2'])] = (r['pred_ag_chain'], r['pred_ab_chain'])

POSES = defaultdict(dict); T1SET={tuple(t) for t in T1}
for r in csv.DictReader(open("results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv")):
    key=(r['assembly_id'][:4], r['chain1'], r['chain2'])
    if key not in T1SET: continue
    try: POSES[key][(int(r['seed']),int(r['sample']))]=float(r['dockq'])
    except: pass

def fast_reps(cif):
    """{chain: np.array([rep_xyz per residue])}, rep=CB (CA fallback), file order — no MMCIF2Dict."""
    with open(cif) as f: lines=f.readlines()
    cols={}; i=0
    for i,ln in enumerate(lines):
        if ln.startswith('_atom_site.'): cols[ln.strip()]=len(cols)
        elif ln.startswith('ATOM'): break
    C=lambda n: cols['_atom_site.'+n]
    ig,ich,iseq,inm=C('group_PDB'),C('auth_asym_id'),C('label_seq_id'),C('label_atom_id')
    ix,iy,iz=C('Cartn_x'),C('Cartn_y'),C('Cartn_z')
    res={}; order=[]
    for ln in lines[i:]:
        if not ln.startswith('ATOM'): continue
        p=ln.split(); key=(p[ich],p[iseq])
        if key not in res: res[key]={}; order.append(key)
        res[key][p[inm]]=(float(p[ix]),float(p[iy]),float(p[iz]))
    out=defaultdict(list)
    for k in order:
        a=res[k]; xyz=a.get('CB',a.get('CA'))
        if xyz: out[k[0]].append(np.array(xyz,dtype=np.float32))
    return {c:np.stack(v) for c,v in out.items()}

def feat_worker(args):
    s,k,cx,ag,ab,agi,abi=args
    cif=DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
    if not cif.exists(): return (s,k,None)
    try:
        R=fast_reps(cif)
        if ag not in R or ab not in R: return (s,k,None)
        A=R[ag][agi]; B=R[ab][abi]
        iu=np.triu_indices(len(A),1)
        intra=np.linalg.norm(A[:,None,:]-A[None,:,:],axis=-1)[iu]
        cross=np.linalg.norm(A[:,None,:]-B[None,:,:],axis=-1).ravel()
        return (s,k,np.concatenate([intra,cross]).astype(np.float32))
    except Exception:
        return (s,k,None)

def process(key):
    cx,c1,c2=key
    if key not in AGAB: return (key,None,"no meta")
    ag,ab=AGAB[key]; poses=sorted(POSES[key].keys())
    ref=None
    for s,k in poses:
        f=DEEP/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
        if f.exists(): ref=fast_reps(f); break
    if not ref or ag not in ref or ab not in ref: return (key,None,"no ref")
    nag,nab=len(ref[ag]),len(ref[ab])
    agi=np.linspace(0,nag-1,min(60,nag)).astype(int); abi=np.linspace(0,nab-1,min(60,nab)).astype(int)
    with Pool(40) as p:
        res=p.map(feat_worker,[(s,k,cx,ag,ab,agi,abi) for s,k in poses])
    rows=[(s,k,f) for s,k,f in res if f is not None]
    if len(rows)<50: return (key,None,f"few({len(rows)})")
    F=np.stack([f for _,_,f in rows]); dq=np.array([POSES[key][(s,k)] for s,k,_ in rows])
    Z=PCA(n_components=min(30,F.shape[0]-1,F.shape[1]), random_state=SEED).fit_transform(StandardScaler().fit_transform(F))
    L=linkage(Z, method='ward')
    minK=None
    for K in KS:
        lab=fcluster(L, K, criterion='maxclust')
        goodmed=False
        for c in np.unique(lab):
            idx=np.where(lab==c)[0]
            cen=Z[idx].mean(0)
            med=idx[np.argmin(((Z[idx]-cen)**2).sum(1))]   # medoid = closest to centroid
            if dq[med]>0.4: goodmed=True; break
        if goodmed: minK=K; break
    # K=100 "good seed cluster" = cluster with the most good docks — its size + stats
    lab = fcluster(L, 100, criterion='maxclust')
    gc = max(np.unique(lab), key=lambda c: (dq[lab == c] > 0.4).sum())
    gi = np.where(lab == gc)[0]; ng = int((dq[gi] > 0.4).sum()); ngt = int((dq > 0.4).sum())
    cen = Z[gi].mean(0); med = gi[np.argmin(((Z[gi] - cen) ** 2).sum(1))]
    return (key, minK, (len(gi), ng, ng / len(gi), ng / max(1, ngt), bool(dq[med] > 0.4)))

if __name__=="__main__":
    print(f"{'iface':>12} {'minK':>5} | K=100 good-seed cluster: {'size':>6} {'nGood':>6} {'purity':>7} {'cover':>6} {'medGood':>7}")
    for t in T1:
        key,minK,k100=process(tuple(t))
        if not isinstance(k100, tuple):
            print(f"{t[0]+' '+t[1]+'-'+t[2]:>12} {str(minK):>5} | SKIP {k100}", flush=True); continue
        size,ng,pur,cov,mg=k100
        print(f"{t[0]+' '+t[1]+'-'+t[2]:>12} {str(minK):>5} | {size:>6} {ng:>6} {pur:>7.3f} {cov:>6.2f} {str(mg):>7}", flush=True)
    print("DONE", flush=True)
