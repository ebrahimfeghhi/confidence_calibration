"""true_iptm_sweep.py — does the '8vzo' finding (true iTM tracks DockQ, but the head predicts it badly)
hold across complexes, or is it one template? For each fg target with a good/bad DockQ range, compute per
pose: DockQ (deep CSV), pred pair_iptm (deep CSV), and TRUE iTM (aligned error in backbone frames vs GT).
Report per-complex corr(true_iTM,DockQ) [is the METRIC ok] and corr(pred_ipTM,true_iTM) [does the HEAD predict it].
"""
import csv, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import forward_grad as fg  # noqa
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
    al=AL.align(a,b)[0]; return sum(1 for u,v in zip(al[0],al[1]) if u==v and u!='-')/min(len(a),len(b))
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
def cross_tm(anchor, partner):
    CAa=np.array([r[3] for r in anchor]); caP,RP=frames(partner)
    return CAa, caP, RP
def true_iTM(pab,pag,gab,gag):
    def one(P_anc,P_par,G_anc,G_par):
        CAaP=np.array([r[3] for r in P_anc]); caPp,RPp=frames(P_par)
        CAaG=np.array([r[3] for r in G_anc]); caGp,RGp=frames(G_par)
        Pp=np.einsum('jba,jib->jia',RPp,CAaP[None]-caPp[:,None])   # R^T @ diff = residue i in frame j (rotation-invariant)
        Gp=np.einsum('jba,jib->jia',RGp,CAaG[None]-caGp[:,None])
        E=np.linalg.norm(Pp-Gp,axis=2)  # [npar, nanc]
        return (1.0/(1.0+(E/d0(E.shape[0]))**2)).mean(0).max()
    return max(one(pab,pag,gab,gag), one(pag,pab,gag,pab if False else gab))
def ab_ag_chains(cx, refpose):
    fg.configure_target(cx); p=parse(str(refpose)); chs=sorted(p.keys())
    off=0; span={}
    for c in chs: span[c]=(off,off+len(p[c])); off+=len(p[c])
    ab=[c for c in chs if span[c][0]>=fg.AB.start and span[c][1]<=fg.AB.stop]
    ag=[c for c in chs if span[c][0]>=fg.C.start and span[c][1]<=fg.C.stop]
    return ab, ag

def main():
    complexes=sys.argv[1:] or ["8ts0","8vzo","8c5h","8jyr","8urf","8z8v"]
    # deep CSV: per complex, per (seed,sample) the best ab-ag dockq + its pair_iptm
    rows=defaultdict(list)
    for r in csv.DictReader(open(DEEP)):
        cx=r['assembly_id'][:4]
        if cx not in complexes: continue
        try: rows[cx].append((int(r['seed']),int(r['sample']),r['chain1'],r['chain2'],float(r['dockq']),float(r['pair_iptm'])))
        except: pass
    print(f"{'cx':>6} {'nposes':>6} {'corr(trueiTM,DockQ)':>19} {'corr(pred,trueiTM)':>18} {'corr(pred,DockQ)':>16}")
    agg=[]
    for cx in complexes:
        gt_f=GTDIR/f"{cx}-assembly1.cif"
        if not gt_f.exists(): print(f"{cx:>6}  (no GT)"); continue
        # find a ref pose that exists
        seeds=sorted({s for s,_,_,_,_,_ in rows[cx]})
        refpose=None
        for s in seeds:
            for k in range(5):
                f=PREDROOT/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
                if f.exists(): refpose=f; break
            if refpose: break
        if not refpose: print(f"{cx:>6}  (no pose)"); continue
        try: ab_ch,ag_ch=ab_ag_chains(cx,refpose)
        except Exception as e: print(f"{cx:>6}  (chain map fail: {e})"); continue
        if not ab_ch or not ag_ch: print(f"{cx:>6}  (no ab/ag split)"); continue
        gt=parse(str(gt_f))
        gab=[max(gt,key=lambda gc:ident(sq(parse(str(refpose))[c]),sq(gt[gc]))) for c in ab_ch]
        gag=[max(gt,key=lambda gc:ident(sq(parse(str(refpose))[c]),sq(gt[gc]))) for c in ag_ch]
        # per (seed,sample): best ab-ag dockq + pair_iptm
        by=defaultdict(lambda:(-1,np.nan))
        for s,k,c1,c2,dqv,pv in rows[cx]:
            key=(s,k)
            if dqv>by[key][0]: by[key]=(dqv,pv)
        allk=sorted(by,key=lambda x:by[x][0])
        sample=set(allk[-12:]) | set(allk[::max(1,len(allk)//30)])   # top-12 + ~30 evenly spread across DockQ
        data=[]
        for s,k in set(sample):
            f=PREDROOT/cx/f"seed_{s}"/"predictions"/f"{cx}_sample_{k}.cif"
            if not f.exists(): continue
            try:
                p=parse(str(f)); Pab=[];Gab=[];Pag=[];Gag=[]
                for pc,gc in zip(ab_ch,gab):
                    a,b=match(p[pc],gt[gc]); Pab+=a;Gab+=b
                for pc,gc in zip(ag_ch,gag):
                    a,b=match(p[pc],gt[gc]); Pag+=a;Gag+=b
                t=true_iTM(Pab,Pag,Gab,Gag)
                dqv,pv=by[(s,k)]; data.append((dqv,pv,t))
            except Exception: continue
        a=np.array([d for d in data if d[1]==d[1]])
        if len(a)<5: print(f"{cx:>6} {len(a):>6}  (too few)"); continue
        from scipy.stats import spearmanr
        c1=spearmanr(a[:,2],a[:,0]).correlation; c2=spearmanr(a[:,1],a[:,2]).correlation; c3=spearmanr(a[:,1],a[:,0]).correlation
        print(f"{cx:>6} {len(a):>6} {c1:>19.2f} {c2:>18.2f} {c3:>16.2f}")
        agg.append((cx,c1,c2,c3))
    if agg:
        A=np.array([[x[1],x[2],x[3]] for x in agg])
        print(f"\nMEAN across {len(agg)} complexes: corr(trueiTM,DockQ)={A[:,0].mean():+.2f}  corr(pred,trueiTM)={A[:,1].mean():+.2f}  corr(pred,DockQ)={A[:,2].mean():+.2f}")

if __name__=="__main__":
    main()
