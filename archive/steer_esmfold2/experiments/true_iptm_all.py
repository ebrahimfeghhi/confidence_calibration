"""true_iptm_all.py — the true-iTM decomposition over ALL complexes with a good/bad DockQ range (not just
fg targets). Identifies the docking interface directly from the deep CSV (the interface with a variable
DockQ range), matches model->GT chains by sequence, computes the TRUE iTM (aligned error in backbone
frames, R^T convention), and per complex reports Spearman corr(trueiTM,DockQ), corr(pred,trueiTM),
corr(pred,DockQ). CPU-parallel over complexes.
"""
import csv, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
from scipy.stats import spearmanr
PREDROOT=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts"
GTDIR=Path("/home/ebrahim/FoldBench/examples/ground_truths")
DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
     'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def parse(cif):
    d=MMCIF2Dict(cif); g=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']; comp=d['_atom_site.label_comp_id']
    at=d['_atom_site.label_atom_id']; sid=d['_atom_site.label_seq_id']
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']
    mdl=d.get('_atom_site.pdbx_PDB_model_num',['1']*len(g)); m0=mdl[0]; res={}
    for i in range(len(g)):
        if g[i]!='ATOM' or mdl[i]!=m0 or comp[i] not in AA3 or at[i] not in ('N','CA','C'): continue
        try: s=int(sid[i])
        except: continue
        res.setdefault((ch[i],s),{'aa':AA3[comp[i]]})[at[i]]=np.array([float(x[i]),float(y[i]),float(z[i])])
    out={}
    for (c,s),r in sorted(res.items(),key=lambda kv:(kv[0][0],kv[0][1])):
        if all(k in r for k in ('N','CA','C')): out.setdefault(c,[]).append((s,r['aa'],r['N'],r['CA'],r['C']))
    return out
def sq(rl): return ''.join(a for _,a,_,_,_ in rl)
AL=PairwiseAligner(); AL.mode='global'; AL.match_score=2; AL.mismatch_score=-1; AL.open_gap_score=-5; AL.extend_gap_score=-.5
def ident(a,b):
    if not a or not b: return 0
    try: al=AL.align(a,b)[0]
    except Exception: return 0
    return sum(1 for u,v in zip(al[0],al[1]) if u==v and u!='-')/max(1,min(len(a),len(b)))
def match(rlP,rlG):
    al=AL.align(sq(rlP),sq(rlG))[0]; iP=iG=0; P=[];G=[]
    for u,v in zip(al[0],al[1]):
        if u!='-' and v!='-' and u==v: P.append(rlP[iP]);G.append(rlG[iG])
        if u!='-':iP+=1
        if v!='-':iG+=1
    return P,G
def frames(rl):
    N=np.array([r[2] for r in rl]); CA=np.array([r[3] for r in rl]); C=np.array([r[4] for r in rl])
    e1=C-CA; e1/=np.linalg.norm(e1,axis=1,keepdims=True)
    v=N-CA; u2=v-(v*e1).sum(1,keepdims=True)*e1; e2=u2/np.linalg.norm(u2,axis=1,keepdims=True)
    return CA, np.stack([e1,e2,np.cross(e1,e2)],axis=2)
def d0(n): return max(0.5, 1.24*(max(n,19)-15)**(1/3)-1.8)
def _dir(P_anc,P_par,G_anc,G_par):
    CAaP=np.array([r[3] for r in P_anc]); caPp,RPp=frames(P_par)
    CAaG=np.array([r[3] for r in G_anc]); caGp,RGp=frames(G_par)
    Pp=np.einsum('jba,jib->jia',RPp,CAaP[None]-caPp[:,None])
    Gp=np.einsum('jba,jib->jia',RGp,CAaG[None]-caGp[:,None])
    E=np.linalg.norm(Pp-Gp,axis=2)
    return (1.0/(1.0+(E/d0(E.shape[0]))**2)).mean(0).max()
def true_iTM(p1,p2,g1,g2): return max(_dir(p1,p2,g1,g2), _dir(p2,p1,g2,g1))

_ROWS=None
def process(cx):
    try:
        rows=_ROWS[cx]                       # (seed,sample,c1,c2,dockq,pair_iptm)
        # pick docking interface = (c1,c2) with a range (min<0.15) and highest max dockq
        byif=defaultdict(list)
        for s,k,c1,c2,dq,pv in rows: byif[(c1,c2)].append((s,k,dq,pv))
        cands=[(iface,v) for iface,v in byif.items()
               if max(x[2] for x in v)>0.4 and min(x[2] for x in v)<0.15 and len(v)>=20]
        if not cands: return (cx,0,None,None,None)
        iface,vv=max(cands,key=lambda kv:max(x[2] for x in kv[1]))
        mc1,mc2=iface     # deep-CSV chains are MODEL chains (not GT)
        gt_f=GTDIR/f"{cx}-assembly1.cif"
        if not gt_f.exists(): return (cx,-1,None,None,None)
        gt=parse(str(gt_f))
        poses=sorted({(s,k) for s,k,_,_ in vv})
        refp=None
        for s,k in poses:
            f=PREDROOT/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
            if f.exists(): refp=parse(str(f)); break
        if not refp: return (cx,-3,None,None,None)
        if mc1 not in refp or mc2 not in refp: return (cx,-2,None,None,None)
        # sequence-match each MODEL interface chain to its GT chain
        g1=max(gt,key=lambda gc:ident(sq(refp[mc1]),sq(gt[gc])))
        g2=max([gc for gc in gt if gc!=g1],key=lambda gc:ident(sq(refp[mc2]),sq(gt[gc])),default=None)
        if g2 is None: return (cx,-4,None,None,None)
        if ident(sq(refp[mc1]),sq(gt[g1]))<0.5 or ident(sq(refp[mc2]),sq(gt[g2]))<0.5: return (cx,-5,None,None,None)
        # stratified sample by DockQ: 10 bands over [0,1], up to 200 evenly-spread per band (~2000/complex)
        bands=defaultdict(list)
        for s,k,dq,pv in vv: bands[min(9,int(dq*10))].append((s,k))
        pick=set()
        for b,items in bands.items():
            step=max(1,len(items)//200); pick.update(items[::step][:200])
        by={(s,k):(dq,pv) for s,k,dq,pv in vv}
        data=[]
        for s,k in pick:
            f=PREDROOT/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
            if not f.exists(): continue
            try:
                p=parse(str(f))
                if mc1 not in p or mc2 not in p: continue
                P1,G1=match(p[mc1],gt[g1]); P2,G2=match(p[mc2],gt[g2])
                if len(P1)<5 or len(P2)<5: continue
                t=true_iTM(P1,P2,G1,G2); dq,pv=by[(s,k)]; data.append((dq,pv,t))
            except Exception: continue
        a=np.array([d for d in data if d[1]==d[1]])
        if len(a)<8: return (cx,len(a),None,None,None)
        c1=spearmanr(a[:,2],a[:,0]).correlation; c2=spearmanr(a[:,1],a[:,2]).correlation; c3=spearmanr(a[:,1],a[:,0]).correlation
        return (cx,len(a),c1,c2,c3)
    except Exception as e:
        return (cx,-9,None,None,None)

def main():
    global _ROWS
    rows=defaultdict(list); best=defaultdict(lambda:-1); worst=defaultdict(lambda:1)
    for r in csv.DictReader(open(DEEP)):
        cx=r['assembly_id'][:4]
        try:
            dq=float(r['dockq']); pv=float(r['pair_iptm'])
            rows[cx].append((int(r['seed']),int(r['sample']),r['chain1'],r['chain2'],dq,pv))
            best[cx]=max(best[cx],dq); worst[cx]=min(worst[cx],dq)
        except: pass
    _ROWS=rows
    cxs=sys.argv[1:] or sorted(rows.keys())   # ALL complexes; process() skips those without an ab-ag range
    print(f"running {len(cxs)} complexes (CPU-parallel)...",flush=True)
    with Pool(16) as p: res=p.map(process, cxs)
    ok=[r for r in res if r[2] is not None]
    print(f"\n{'cx':>6} {'n':>4} {'trueiTM~DockQ':>13} {'pred~trueiTM':>12} {'pred~DockQ':>10}")
    for cx,n,c1,c2,c3 in sorted(res,key=lambda r:(r[4] if r[4] is not None else 99)):
        if c1 is None: continue
        print(f"{cx:>6} {n:>4} {c1:>13.2f} {c2:>12.2f} {c3:>10.2f}")
    if ok:
        A=np.array([[r[2],r[3],r[4]] for r in ok])
        print(f"\n=== {len(ok)} complexes ===")
        print(f"MEAN  trueiTM~DockQ={np.nanmean(A[:,0]):+.2f}  pred~trueiTM={np.nanmean(A[:,1]):+.2f}  pred~DockQ={np.nanmean(A[:,2]):+.2f}")
        print(f"MEDIAN trueiTM~DockQ={np.nanmedian(A[:,0]):+.2f}  pred~trueiTM={np.nanmedian(A[:,1]):+.2f}  pred~DockQ={np.nanmedian(A[:,2]):+.2f}")
        print(f"pred~DockQ works (>0.3): {int((A[:,2]>0.3).sum())}/{len(ok)}   fails (<0): {int((A[:,2]<0).sum())}/{len(ok)}")
        skipped=[r for r in res if r[2] is None]
        print(f"skipped: {len(skipped)} (no GT / no range / parse fail)")

if __name__=="__main__":
    main()
