"""why_good_8vzo.py — why are 8vzo's rare good samples good? Correct epitope, or a small antigen shift?

Superpose each pose's ANTIBODY onto the ground truth (Kabsch on matched CA), apply the same transform to
the antigen, and measure the antigen CA-RMSD to the GT antigen (= how far the docked antigen is from where
it belongs). Also the GT-epitope overlap. Then plot DockQ vs antigen-RMSD across a DockQ-spanning sample:
- good poses at LOW rmsd  -> they dock the correct surface (correct epitope aligned to GT)
- large rmsd gap good<->bad -> real reorientation, not a small shift.

  python steer/experiments/why_good_8vzo.py
"""
import csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
GT="/home/ebrahim/FoldBench/examples/ground_truths/8vzo-assembly1.cif"
PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts/8vzo"
DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
     'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def parse(cif):
    d=MMCIF2Dict(cif); g=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']; comp=d['_atom_site.label_comp_id']
    at=d['_atom_site.label_atom_id']; sid=d['_atom_site.label_seq_id']
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']
    mdl=d.get('_atom_site.pdbx_PDB_model_num',['1']*len(g)); m0=mdl[0]; res={}
    for i in range(len(g)):
        if g[i]!='ATOM' or mdl[i]!=m0 or comp[i] not in AA3 or at[i]!='CA': continue
        try: s=int(sid[i])
        except: continue
        res.setdefault(ch[i],[]).append((s,AA3[comp[i]],np.array([float(x[i]),float(y[i]),float(z[i])])))
    return {c:sorted(v) for c,v in res.items()}
def seq(rl): return ''.join(a for _,a,_ in rl)
AL=PairwiseAligner(); AL.mode='global'; AL.match_score=2; AL.mismatch_score=-1; AL.open_gap_score=-5; AL.extend_gap_score=-.5
def ident(a,b):
    if not a or not b: return 0.0
    al=AL.align(a,b)[0]; return sum(1 for u,v in zip(al[0],al[1]) if u==v and u!='-')/min(len(a),len(b))
def matched_ca(rlP,rlG):
    al=AL.align(seq(rlP),seq(rlG))[0]; iP=iG=0; P=[]; G=[]
    for u,v in zip(al[0],al[1]):
        if u!='-' and v!='-' and u==v: P.append(rlP[iP][2]); G.append(rlG[iG][2])
        if u!='-': iP+=1
        if v!='-': iG+=1
    return np.array(P), np.array(G)
def kabsch(P,Q):
    Pc=P-P.mean(0); Qc=Q-Q.mean(0); H=Pc.T@Qc; U,_,Vt=np.linalg.svd(H); D=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,D])@U.T; return R, Q.mean(0)-R@P.mean(0)

def main():
    gt=parse(GT)
    # map roles: predicted A = antigen, B,C = antibody. Find GT chains by sequence identity to a ref pose.
    ref=parse(str(PRED/"seed_166"/"predictions"/"8vzo_sample_3.cif"))
    ag_p='A'; ab_p=[c for c in ref if c!='A']
    def best_gt(pseq): return max(gt, key=lambda gc: ident(pseq, seq(gt[gc])))
    gt_ag=best_gt(seq(ref[ag_p])); gt_ab=[best_gt(seq(ref[c])) for c in ab_p]
    print(f"chain map: antigen pred A -> GT {gt_ag};  antibody pred {ab_p} -> GT {gt_ab}",flush=True)
    GTab=np.vstack([gt[c][:,None] if False else np.array([r[2] for r in gt[c]]) for c in gt_ab])
    GTag_rl=gt[gt_ag]

    # GT epitope (GT antigen residues within 8A of GT antibody)
    GTag_xyz=np.array([r[2] for r in GTag_rl])
    gt_epi=set(GTag_rl[i][0] for i in range(len(GTag_rl)) if (np.linalg.norm(GTab-GTag_xyz[i],axis=1)<8).any())

    # sample poses spanning DockQ
    dq={}
    for r in csv.DictReader(open(DEEP)):
        if r['assembly_id'][:4]!='8vzo': continue
        try: k=(int(r['seed']),int(r['sample'])); v=float(r['dockq'])
        except: continue
        dq[k]=max(dq.get(k,0.0),v)
    good=[(166,3),(809,4),(201,2),(246,3),(981,0),(95,1),(946,3)]
    allk=sorted(dq, key=lambda k:dq[k])
    sample=set(good) | set(allk[::len(allk)//60]) | set(allk[-20:])   # good + spread + top

    rows=[]
    for seed,samp in sorted(sample):
        f=PRED/f"seed_{seed}"/"predictions"/f"8vzo_sample_{samp}.cif"
        if not f.exists(): continue
        p=parse(str(f))
        try:
            # superpose antibody pred->GT, apply to antigen, antigen rmsd
            Pab=[];Gab=[]
            for pc,gc in zip(ab_p,gt_ab):
                a,b=matched_ca(p[pc],gt[gc]); Pab.append(a);Gab.append(b)
            Pab=np.vstack(Pab);Gab=np.vstack(Gab)
            R,t=kabsch(Pab,Gab)
            Pag,Gag=matched_ca(p[ag_p],GTag_rl)
            rmsd=np.sqrt((((R@Pag.T).T+t-Gag)**2).sum(1).mean())
            # pred epitope in pred frame -> map to GT residue ids via antigen alignment
            al=AL.align(seq(p[ag_p]),seq(GTag_rl))[0]; iP=iG=0; pmap={}
            for u,v in zip(al[0],al[1]):
                if u!='-' and v!='-': pmap[p[ag_p][iP][0]]=GTag_rl[iG][0]
                if u!='-': iP+=1
                if v!='-': iG+=1
            pab=np.vstack([np.array([r[2] for r in p[c]]) for c in ab_p])
            p_epi=set(pmap.get(p[ag_p][i][0]) for i in range(len(p[ag_p]))
                      if (np.linalg.norm(pab-p[ag_p][i][2],axis=1)<8).any())
            p_epi.discard(None); jac=len(p_epi&gt_epi)/max(len(p_epi|gt_epi),1)
            rows.append((dq[(seed,samp)], rmsd, jac, seed, samp))
        except Exception as e:
            continue
    rows.sort()
    print(f"\n{'DockQ':>6} {'antigenRMSD':>11} {'GTepitopeJac':>12}  (sampled {len(rows)} poses)")
    for d,rm,j,s,k in rows:
        tag=" <-- GOOD" if (s,k) in good else ""
        print(f"{d:>6.2f} {rm:>11.1f} {j:>12.2f}{tag}")
    gr=[r for r in rows if (r[3],r[4]) in good]; br=[r for r in rows if r[0]<0.1]
    if gr and br:
        print(f"\ngood poses: antigenRMSD {np.mean([r[1] for r in gr]):.1f}A, GT-epitope Jac {np.mean([r[2] for r in gr]):.2f}")
        print(f"bad poses:  antigenRMSD {np.mean([r[1] for r in br]):.1f}A, GT-epitope Jac {np.mean([r[2] for r in br]):.2f}")

    fig,ax=plt.subplots(figsize=(6.5,5))
    d=[r[0] for r in rows]; rm=[r[1] for r in rows]; isg=[(r[3],r[4]) in good for r in rows]
    ax.scatter([x for x,g in zip(d,isg) if not g],[y for y,g in zip(rm,isg) if not g],s=18,c="#888",label="pose")
    ax.scatter([x for x,g in zip(d,isg) if g],[y for y,g in zip(rm,isg) if g],s=45,c="#c33",label="good sample")
    ax.set_xlabel("DockQ"); ax.set_ylabel("antigen RMSD to GT (A, after antibody superposition)")
    ax.set_title("8vzo: is a good DockQ = correctly-placed antigen?"); ax.legend(); ax.grid(alpha=.3)
    out=Path("steer/experiments/why_good_8vzo.png"); fig.tight_layout(); fig.savefig(out,dpi=130)
    print(f"\nwrote {out}",flush=True)

if __name__=="__main__":
    main()
