"""pose_shift_8vzo.py — how big is the antibody move between a GOOD (Fnat 1.0) and MEDIOCRE (Fnat 0) 8vzo
pose? Superpose the two poses' ANTIGENS (same protein), apply that transform to the mediocre pose's
antibody, then measure how far the antibody sits from the good pose's antibody: rigid rotation angle +
centroid translation + CA-RMSD. Small numbers => Fnat flips on a nudge; large => a real reorientation.
"""
import numpy as np
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts/8vzo"
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

def main():
    good=parse(str(PRED/"seed_166"/"predictions"/"8vzo_sample_3.cif"))
    ab_ch=[c for c in good if c!='A']
    mids=[("402",2),("229",0),("554",2),("95",1)]  # 95 s1 = DockQ 0.49, near-good; others DockQ ~0.12
    print("antibody move from GOOD(166 s3, Fnat 1.0) after aligning antigens:")
    print(f"{'mid':>10} {'antigenFoldRMSD':>14} {'ab_rotDeg':>9} {'ab_transA':>9} {'ab_RMSD':>8}")
    for seed,samp in mids:
        m=parse(str(PRED/f"seed_{seed}"/"predictions"/f"8vzo_sample_{samp}.cif"))
        Pag_m,Pag_g=matched(m['A'],good['A'])          # mid antigen -> good antigen
        R,t=kabsch(Pag_m,Pag_g)
        agrmsd=np.sqrt((((R@Pag_m.T).T+t-Pag_g)**2).sum(1).mean())
        # antibody matched CAs (per chain), transform mid ab into good frame
        Gab=[];Mab=[]
        for c in ab_ch:
            if c in m:
                a,b=matched(m[c],good[c]); Mab.append(a); Gab.append(b)
        Gab=np.vstack(Gab); Mab=np.vstack(Mab)
        Mab_g=(R@Mab.T).T+t                             # mid antibody in good's antigen frame
        ab_rmsd=np.sqrt(((Gab-Mab_g)**2).sum(1).mean())
        trans=np.linalg.norm(Gab.mean(0)-Mab_g.mean(0))
        Rab,_=kabsch(Mab_g,Gab); ang=np.degrees(np.arccos(np.clip((np.trace(Rab)-1)/2,-1,1)))
        print(f"{seed+' s'+str(samp):>10} {agrmsd:>14.1f} {ang:>9.0f} {trans:>9.1f} {ab_rmsd:>8.1f}")

if __name__=="__main__":
    main()
