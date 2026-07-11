"""cluster_pose_8vzo.py — re-cluster 8vzo diffusion samples on an ORIENTATION-AWARE feature: superpose each
pose's antibody onto a common reference antibody, then take the antigen CA coords in that frame. This
encodes the full relative dock (position + orientation), which the min-distance/epitope feature threw away.
Question: do the ~7 good poses now separate as distinct outliers instead of being buried in cluster 1?
"""
import sys, csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts/8vzo"
DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
     'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def parse(cif):
    d=MMCIF2Dict(cif); g=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']; comp=d['_atom_site.label_comp_id']
    at=d['_atom_site.label_atom_id']; sid=d['_atom_site.label_seq_id']
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']; res={}
    for i in range(len(g)):
        if g[i]!='ATOM' or comp[i] not in AA3 or at[i]!='CA': continue
        try: s=int(sid[i])
        except: continue
        res.setdefault(ch[i],[]).append((s,AA3[comp[i]],np.array([float(x[i]),float(y[i]),float(z[i])])))
    return {c:sorted(v) for c,v in res.items()}
def sq(rl): return ''.join(a for _,a,_ in rl)
AL=PairwiseAligner(); AL.mode='global'; AL.match_score=2; AL.mismatch_score=-1; AL.open_gap_score=-5; AL.extend_gap_score=-.5
def matched(rlA,rlB):
    al=AL.align(sq(rlA),sq(rlB))[0]; iA=iB=0; A=[];B=[]
    for u,v in zip(al[0],al[1]):
        if u!='-' and v!='-' and u==v: A.append(rlA[iA][2]); B.append(rlB[iB][2])
        if u!='-': iA+=1
        if v!='-': iB+=1
    return np.array(A),np.array(B)
def kabsch(P,Q):
    Pc=P-P.mean(0); Qc=Q-Q.mean(0); H=Pc.T@Qc; U,_,Vt=np.linalg.svd(H); D=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,D])@U.T; return R, Q.mean(0)-R@P.mean(0)
_REF=None
def feat_one(sk):
    seed,samp=sk
    cif=PRED/f"seed_{seed}"/"predictions"/f"8vzo_sample_{samp}.cif"
    if not cif.exists(): return sk,None
    try:
        p=parse(str(cif)); Pab=[];Rab=[]
        for c in _REF['ab']:
            if c not in p: return sk,None
            a,b=matched(p[c],_REF['ab'][c]); Pab.append(a);Rab.append(b)
        Pab=np.vstack(Pab);Rab=np.vstack(Rab); R,t=kabsch(Pab,Rab)
        ag=np.array([r[2] for r in p['A']])
        if ag.shape[0]!=_REF['agn']: return sk,None
        return sk, ((R@ag.T).T+t).reshape(-1).astype(np.float32)
    except Exception: return sk,None
def main():
    global _REF
    ref=parse(str(PRED/"seed_166"/"predictions"/"8vzo_sample_3.cif"))
    _REF={'ab':{c:ref[c] for c in ref if c!='A'},'agn':len(ref['A'])}
    tasks=[(s,k) for s in range(1024) for k in range(5)]
    print(f"=== 8vzo orientation-aware clustering: {len(tasks)} poses, feature=antigen coords in antibody frame ({_REF['agn']*3}d) ===",flush=True)
    with Pool(16) as p: res=p.map(feat_one,tasks)
    keys=[sk for sk,f in res if f is not None]; X=np.stack([f for sk,f in res if f is not None])
    print(f"parsed {len(keys)}, dim {X.shape[1]}",flush=True)
    dq={}
    for r in csv.DictReader(open(DEEP)):
        if r['assembly_id'][:4]!='8vzo': continue
        try: dq[(int(r['seed']),int(r['sample']))]=max(dq.get((int(r['seed']),int(r['sample'])),0.0),float(r['dockq']))
        except: pass
    lab=np.array([dq.get(sk,np.nan) for sk in keys])
    Xp=PCA(n_components=30).fit_transform(StandardScaler().fit_transform(X))
    cl=HDBSCAN(min_cluster_size=20).fit_predict(Xp)
    good=[(166,3),(809,4),(201,2),(246,3),(981,0),(95,1),(946,3)]
    kidx={sk:i for i,sk in enumerate(keys)}
    from collections import Counter
    print(f"\nclusters: {len(set(cl)-{-1})}  noise: {int((cl==-1).sum())}")
    print("where do the good poses land? (cluster -1 = noise/outlier)")
    for gk in good:
        if gk in kidx:
            i=kidx[gk]; c=cl[i]
            sz=int((cl==c).sum()); md=np.nanmean(lab[cl==c]) if c!=-1 else float('nan')
            print(f"  {gk} DockQ {lab[i]:.2f} -> cluster {c}  (cluster size {sz}, mean DockQ {md:.3f})"
                  + ("  <-- DISTINCT/outlier" if c==-1 or sz<15 else "  <-- buried"))
    # how isolated are good poses? mean dist to nearest good vs nearest any
    fig,ax=plt.subplots(figsize=(6.8,5.6))
    o=np.argsort(np.nan_to_num(lab,nan=-1))
    ax.scatter(Xp[o,0],Xp[o,1],c=np.nan_to_num(lab[o],nan=-0.05),cmap="viridis",s=8,vmin=0,vmax=1)
    gi=[kidx[g] for g in good if g in kidx]
    ax.scatter(Xp[gi,0],Xp[gi,1],s=90,facecolors='none',edgecolors='red',linewidths=1.8,label='good poses')
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    ax.set_title("8vzo orientation-aware (antigen coords in antibody frame), color=DockQ")
    out=Path("steer/experiments/cluster_pose_8vzo.png"); fig.tight_layout(); fig.savefig(out,dpi=130)
    print(f"\nwrote {out}",flush=True)
if __name__=="__main__":
    main()
