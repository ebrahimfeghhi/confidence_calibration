"""true_iptm_8vzo.py — is ipTM's DockQ-blindness a METRIC problem or a PREDICTION problem?

Compute the TRUE interface TM (what ipTM tries to predict) by using the ACTUAL aligned error instead of the
predicted PAE: for each cross-chain pair (i,j), express residue i's CA in residue j's backbone frame, in
both the prediction and the GT, and take the distance = true aligned error. Then the same ipTM formula
(TM contribution f(e)=1/(1+(e/d0)^2), max-anchor over one chain, mean over the other). Compare across poses:
  corr(true_iTM, DockQ)      -> is the metric itself a good DockQ proxy?  (a)
  corr(pred_ipTM, true_iTM)  -> does the confidence head predict it well? (b)
"""
import csv, json, numpy as np
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
GT="/home/ebrahim/FoldBench/examples/ground_truths/8vzo-assembly1.cif"
PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts/8vzo"
DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
     'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def parse(cif):
    """{chain: [(resid, aa, N,CA,C)]} backbone atoms."""
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
def frames(rl):
    """CA origins [n,3] and rotation R [n,3,3] (columns e1,e2,e3) from N,CA,C."""
    N=np.array([r[2] for r in rl]); CA=np.array([r[3] for r in rl]); C=np.array([r[4] for r in rl])
    e1=C-CA; e1/=np.linalg.norm(e1,axis=1,keepdims=True)
    v=N-CA; u2=v-(v*e1).sum(1,keepdims=True)*e1; e2=u2/np.linalg.norm(u2,axis=1,keepdims=True)
    e3=np.cross(e1,e2); R=np.stack([e1,e2,e3],axis=2)  # [n,3,3]
    return CA, R
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
def d0(n): return max(0.5, 1.24*(max(n,19)-15)**(1/3)-1.8)
def tm(e,dd): return 1.0/(1.0+(e/dd)**2)

def local(rl_ref_frames_CA, rl_ref_frames_R, CA_i):
    # position of each CA_i in each frame j -> [nj, ni, 3]
    diff=CA_i[None,:,:]-rl_ref_frames_CA[:,None,:]      # [nj, ni, 3]
    return np.einsum('jab,jib->jia', rl_ref_frames_R, diff)  # R_j^T @ diff  -> [nj, ni, 3]

def true_iptm(predP_ab, predP_ag, gtP_ab, gtP_ag):
    """ab anchors, ag partners: true aligned error of ab_i in ag_j frame, pred vs GT; ipTM = max_i mean_j tm."""
    CAp_ab=np.array([r[3] for r in predP_ab]); CAg_ab=np.array([r[3] for r in gtP_ab])
    caP_ag,RP_ag=frames(predP_ag); caG_ag,RG_ag=frames(gtP_ag)
    Ppos=local(caP_ag,RP_ag,CAp_ab)   # [nag, nab, 3]  ab_i in pred ag_j frame
    Gpos=local(caG_ag,RG_ag,CAg_ab)
    E=np.linalg.norm(Ppos-Gpos,axis=2)   # [nag, nab] true aligned error
    F=tm(E, d0(E.shape[0]))              # partner set = antigen
    return F.mean(0).max()               # max over ab anchor (axis1) of mean over ag partner (axis0)

def main():
    gt=parse(GT); ref=parse(str(PRED/"seed_166"/"predictions"/"8vzo_sample_3.cif"))
    ag_p='A'; ab_p=[c for c in ref if c!='A']
    gt_ag=max(gt,key=lambda gc:ident(sq(ref[ag_p]),sq(gt[gc])))
    gt_ab=[max(gt,key=lambda gc:ident(sq(ref[c]),sq(gt[gc]))) for c in ab_p]
    dq={}; pip={}
    for r in csv.DictReader(open(DEEP)):
        if r['assembly_id'][:4]!='8vzo': continue
        try:
            k=(int(r['seed']),int(r['sample'])); dq[k]=max(dq.get(k,0.0),float(r['dockq']))
            pip[k]=max(pip.get(k,0.0),float(r['pair_iptm']))
        except: pass
    good=[(166,3),(809,4),(201,2),(246,3),(981,0),(95,1),(946,3)]
    allk=sorted(dq,key=lambda k:dq[k])
    poses=good+[k for k in allk if 0.08<=dq[k]<=0.2][:6]+allk[:6]
    print(f"{'seed':>5} {'samp':>4} {'DockQ':>6} {'pred_ipTM':>9} {'true_iTM':>9}")
    rows=[]
    for seed,samp in poses:
        cif=PRED/f"seed_{seed}"/"predictions"/f"8vzo_sample_{samp}.cif"
        if not cif.exists(): continue
        p=parse(str(cif))
        try:
            # match ab (concat over chains) and ag to GT
            Pab=[];Gab=[]
            for pc,gc in zip(ab_p,gt_ab):
                a,b=match(p[pc],gt[gc]); Pab+=a; Gab+=b
            Pag,Gag=match(p[ag_p],gt[gt_ag])
            t=true_iptm(Pab,Pag,Gab,Gag)
            rows.append((dq[(seed,samp)], pip.get((seed,samp),np.nan), t, seed, samp))
            print(f"{seed:>5} {samp:>4} {dq[(seed,samp)]:>6.2f} {pip.get((seed,samp),float('nan')):>9.3f} {t:>9.3f}")
        except Exception as e: print(f"{seed} {samp}: {e}")
    a=np.array([(d,p,t) for d,p,t,_,_ in rows if p==p])
    if len(a)>3:
        print(f"\ncorr(true_iTM, DockQ)     = {np.corrcoef(a[:,2],a[:,0])[0,1]:+.2f}   <- (a) is the METRIC a good DockQ proxy?")
        print(f"corr(pred_ipTM, true_iTM) = {np.corrcoef(a[:,1],a[:,2])[0,1]:+.2f}   <- (b) does the HEAD predict it?")
        print(f"corr(pred_ipTM, DockQ)    = {np.corrcoef(a[:,1],a[:,0])[0,1]:+.2f}   (the observed failure)")

if __name__=="__main__":
    main()
