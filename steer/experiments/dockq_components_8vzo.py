"""dockq_components_8vzo.py — decompose DockQ into Fnat / iRMSD / LRMSD for good vs mediocre vs wrong 8vzo
poses, to see WHICH component distinguishes good (0.6) from mediocre (0.15). CA-based approximations
(no heavy atoms / OST available), so absolute values are approximate — but the relative pattern and the
reconstructed DockQ (sanity-checked vs the true DockQ) tell us what's moving.

  Fnat  = frac of native (antibody-i, antigen-j) CA-contacts (<8A) recovered
  iRMSD = CA RMSD of GT-interface residues after superposing them
  LRMSD = antigen CA RMSD after superposing the antibody
  DockQ ~ (Fnat + 1/(1+(iRMSD/1.5)^2) + 1/(1+(LRMSD/8.5)^2))/3
"""
import csv, numpy as np
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
GT="/home/ebrahim/FoldBench/examples/ground_truths/8vzo-assembly1.cif"
PRED=Path.home()/"esm/foldbench_predictions/esmfold2fast_tts/8vzo"
DEEP="results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
     'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
CUT=8.0
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
def sq(rl): return ''.join(a for _,a,_ in rl)
AL=PairwiseAligner(); AL.mode='global'; AL.match_score=2; AL.mismatch_score=-1; AL.open_gap_score=-5; AL.extend_gap_score=-.5
def ident(a,b):
    if not a or not b: return 0
    al=AL.align(a,b)[0]; return sum(1 for u,v in zip(al[0],al[1]) if u==v and u!='-')/min(len(a),len(b))
def resmap(rlP,rlG):
    """pred resid -> GT resid via alignment; also matched CA arrays."""
    al=AL.align(sq(rlP),sq(rlG))[0]; iP=iG=0; mp={}; P=[]; G=[]
    for u,v in zip(al[0],al[1]):
        if u!='-' and v!='-':
            mp[rlP[iP][0]]=rlG[iG][0]
            if u==v: P.append(rlP[iP][2]); G.append(rlG[iG][2])
        if u!='-': iP+=1
        if v!='-': iG+=1
    return mp, np.array(P), np.array(G)
def kabsch(P,Q):
    Pc=P-P.mean(0); Qc=Q-Q.mean(0); H=Pc.T@Qc; U,_,Vt=np.linalg.svd(H); D=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,D])@U.T; return R, Q.mean(0)-R@P.mean(0)
def contacts(ab_rl, ag_rl):
    A=np.array([r[2] for r in ab_rl]); G=np.array([r[2] for r in ag_rl])
    Dm=np.linalg.norm(A[:,None]-G[None],axis=-1); ii,jj=np.where(Dm<CUT)
    return set((ab_rl[i][0], ag_rl[j][0]) for i,j in zip(ii,jj))

def main():
    gt=parse(GT); ref=parse(str(PRED/"seed_166"/"predictions"/"8vzo_sample_3.cif"))
    ag_p='A'; ab_p=[c for c in ref if c!='A']
    gt_ag=max(gt,key=lambda gc:ident(sq(ref[ag_p]),sq(gt[gc])))
    gt_ab=[max(gt,key=lambda gc:ident(sq(ref[c]),sq(gt[gc]))) for c in ab_p]
    GTab_rl=[r for c in gt_ab for r in gt[c]]; GTag_rl=gt[gt_ag]
    gt_ct=contacts(GTab_rl, GTag_rl)                       # native contacts in GT resid space
    # GT interface residue CAs (both sides) for iRMSD
    gt_if_ab={a for a,_ in gt_ct}; gt_if_ag={b for _,b in gt_ct}

    dq={}
    for r in csv.DictReader(open(DEEP)):
        if r['assembly_id'][:4]!='8vzo': continue
        try: dq[(int(r['seed']),int(r['sample']))]=max(dq.get((int(r['seed']),int(r['sample'])),0.0),float(r['dockq']))
        except: pass
    allk=sorted(dq,key=lambda k:dq[k])
    good=[(166,3)]; mid=[k for k in allk if 0.12<=dq[k]<=0.20][:3]; wrong=allk[:2]
    print(f"{'seed':>5} {'samp':>4} {'trueDockQ':>9} | {'Fnat':>5} {'iRMSD':>6} {'LRMSD':>6} | {'reconDockQ':>10}  tier")
    for tier,poses in [("good",good),("mid",mid),("wrong",wrong)]:
        for seed,samp in poses:
            f=PRED/f"seed_{seed}"/"predictions"/f"8vzo_sample_{samp}.cif"
            if not f.exists(): continue
            p=parse(str(f))
            # maps pred->GT resid per chain
            mp_ag,_,_=resmap(p[ag_p],GTag_rl)
            # antibody: match each pred ab chain to a GT ab chain
            ab_maps={}; Pab=[];Gab=[]
            for pc,gc in zip(ab_p,gt_ab):
                m,PA,GA=resmap(p[pc],gt[gc]); ab_maps[pc]=(m,gc); Pab.append(PA);Gab.append(GA)
            Pab=np.vstack(Pab);Gab=np.vstack(Gab)
            # pred contacts, mapped to GT resid space
            pab_rl=[r for c in ab_p for r in p[c]]
            pred_ct_raw=contacts(pab_rl, p[ag_p])
            # need pred ab resid -> GT resid; ab residue ids may collide across chains, so map via chain
            # rebuild pred ab contacts using a global resid->gt map (assume unique resid per ab chain concatenation is fine since GT contact set is in GT numbering)
            gmap_ab={}
            for pc in ab_p:
                for pr in p[pc]:
                    if pr[0] in ab_maps[pc][0]: gmap_ab[pr[0]]=ab_maps[pc][0][pr[0]]
            pred_ct=set((gmap_ab.get(a), mp_ag.get(b)) for a,b in pred_ct_raw)
            pred_ct.discard((None,None))
            fnat=len(pred_ct & gt_ct)/max(len(gt_ct),1)
            # LRMSD: superpose antibody, apply to antigen
            R,t=kabsch(Pab,Gab); _,Pag,Gag=resmap(p[ag_p],GTag_rl)
            lrmsd=np.sqrt((((R@Pag.T).T+t-Gag)**2).sum(1).mean())
            # iRMSD: superpose GT-interface residues (pred vs GT), rmsd
            if_pairsP=[];if_pairsG=[]
            invmap={v:k for pc in ab_p for k,v in ab_maps[pc][0].items()}
            for pc in ab_p:
                for pr in p[pc]:
                    gid=ab_maps[pc][0].get(pr[0])
                    if gid in gt_if_ab:
                        gres=[r for r in gt[ab_maps[pc][1]] if r[0]==gid]
                        if gres: if_pairsP.append(pr[2]); if_pairsG.append(gres[0][2])
            invag={v:k for k,v in mp_ag.items()}
            for pr in p[ag_p]:
                gid=mp_ag.get(pr[0])
                if gid in gt_if_ag:
                    gres=[r for r in GTag_rl if r[0]==gid]
                    if gres: if_pairsP.append(pr[2]); if_pairsG.append(gres[0][2])
            if len(if_pairsP)>=3:
                Pi=np.array(if_pairsP);Gi=np.array(if_pairsG); Ri,ti=kabsch(Pi,Gi)
                irmsd=np.sqrt((((Ri@Pi.T).T+ti-Gi)**2).sum(1).mean())
            else: irmsd=float('nan')
            recon=(fnat + 1/(1+(irmsd/1.5)**2) + 1/(1+(lrmsd/8.5)**2))/3
            print(f"{seed:>5} {samp:>4} {dq[(seed,samp)]:>9.2f} | {fnat:>5.2f} {irmsd:>6.1f} {lrmsd:>6.1f} | {recon:>10.2f}  {tier}")

if __name__=="__main__":
    main()
