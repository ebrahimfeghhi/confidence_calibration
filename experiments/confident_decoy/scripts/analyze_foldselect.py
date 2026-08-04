import json,glob,os,re,statistics as st
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gemmi
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
chainmap=json.load(open(f"{SC}/oracle_chainmap.json"))     # {pdb:{qi:native_letter}}
antigen=json.load(open(f"{SC}/agonly_antigen_idx.json"))
SEEDS=range(101,151)
def cached_lddt(pdb,qi,s,k):
    f=f"{SC}/ost_oracle/{pdb}_{qi}_{s}_{k}.json"
    try:
        d=json.load(open(f)); return d.get("lddt"),d.get("_mc")
    except: return None,None
def pose_feats(pdb,qi,s,k):
    ld,mc=cached_lddt(pdb,qi,s,k)
    if ld is None or mc is None: return None
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    try: stm=gemmi.read_structure(cif)
    except: return None
    ch=next((c for c in stm[0] if c.name==mc),None)
    if ch is None: return None
    bs=[]; ca=[]
    for r in ch:
        a=r.find_atom("CA","*") if hasattr(r,'find_atom') else None
        if a is None:
            a=next((at for at in r if at.name=="CA"),None)
        if a is None: continue
        bs.append(a.b_iso); ca.append([a.pos.x,a.pos.y,a.pos.z])
    if not ca: return None
    return dict(lddt=ld,plddt=float(np.mean(bs)),ca=np.array(ca),s=s,k=k)

def kabsch_align(P,Q):  # align P onto Q, return rmsd
    Pc=P-P.mean(0); Qc=Q-Q.mean(0)
    V,S,Wt=np.linalg.svd(Pc.T@Qc)
    d=np.sign(np.linalg.det(V@Wt)); D=np.diag([1,1,d])
    R=V@D@Wt; Pr=Pc@R
    return np.sqrt(((Pr-Qc)**2).sum(1).mean())

def analyze_chain(pdb,qi):
    feats=[]
    for s in SEEDS:
        for k in range(5):
            f=pose_feats(pdb,qi,s,k)
            if f: feats.append(f)
    if len(feats)<10: return None
    L=st.mode([len(f["ca"]) for f in feats])
    feats=[f for f in feats if len(f["ca"])==L]   # common length for superposition
    n=len(feats)
    lddt=np.array([f["lddt"] for f in feats]); plddt=np.array([f["plddt"] for f in feats])
    CA=np.stack([f["ca"] for f in feats])         # (n,L,3)
    # consensus: mean structure (superpose all to ref0), then centrality = -rmsd to mean
    ref=CA[0]-CA[0].mean(0)
    aligned=[]
    for i in range(n):
        Pc=CA[i]-CA[i].mean(0)
        V,S,Wt=np.linalg.svd(Pc.T@ref); d=np.sign(np.linalg.det(V@Wt)); R=V@np.diag([1,1,d])@Wt
        aligned.append(Pc@R)
    aligned=np.stack(aligned); mean_struct=aligned.mean(0)
    rmsd_to_mean=np.array([np.sqrt(((aligned[i]-mean_struct)**2).sum(1).mean()) for i in range(n)])
    # selectors -> index
    i_oracle=int(np.argmax(lddt))
    i_plddt=int(np.argmax(plddt))
    i_cons=int(np.argmin(rmsd_to_mean))
    # spearman-ish: correlation of plddt vs lddt, and centrality vs lddt
    def corr(a,b):
        if np.std(a)<1e-9 or np.std(b)<1e-9: return 0.0
        ar=np.argsort(np.argsort(a)); br=np.argsort(np.argsort(b))
        return float(np.corrcoef(ar,br)[0,1])
    role="ag" if int(qi) in antigen[pdb] else "ab"
    return dict(pdb=pdb,qi=qi,role=role,n=n,
        oracle=float(lddt[i_oracle]), plddt_pick=float(lddt[i_plddt]), cons_pick=float(lddt[i_cons]),
        pool_max=float(lddt.max()), pool_mean=float(lddt.mean()), pool_min=float(lddt.min()),
        sp_plddt=corr(plddt,lddt), sp_cons=corr(-rmsd_to_mean,lddt))

jobs=[(pdb,qi) for pdb,d in chainmap.items() for qi in d]
print("chains:",len(jobs),flush=True)
rows=[]
with ThreadPoolExecutor(max_workers=12) as ex:
    for r in ex.map(lambda j: analyze_chain(*j),jobs):
        if r: rows.append(r); print(f"{r['pdb']} q{r['qi']}({r['role']}) oracle {r['oracle']:.3f} plddt {r['plddt_pick']:.3f} cons {r['cons_pick']:.3f} | corr(plddt,lddt) {r['sp_plddt']:+.2f} corr(cons,lddt) {r['sp_cons']:+.2f}",flush=True)
json.dump(rows,open(f"{SC}/foldselect_rows.json","w"),indent=1)
# aggregate
def agg(key):
    v=[r[key] for r in rows]; return f"{np.mean(v):.3f}"
print("\n=== AGG (mean lDDT of picked fold, over %d chains) ==="%len(rows))
print(f"oracle(max lDDT)   : {agg('oracle')}")
print(f"pLDDT-pick         : {agg('plddt_pick')}   regret {np.mean([r['oracle']-r['plddt_pick'] for r in rows]):.3f}")
print(f"consensus-pick     : {agg('cons_pick')}   regret {np.mean([r['oracle']-r['cons_pick'] for r in rows]):.3f}")
print(f"pool mean          : {agg('pool_mean')}")
print(f"\nmean rank-corr(pLDDT, lDDT)     : {np.mean([r['sp_plddt'] for r in rows]):+.3f}")
print(f"mean rank-corr(centrality, lDDT): {np.mean([r['sp_cons'] for r in rows]):+.3f}")
print("ANALYZE_DONE")
