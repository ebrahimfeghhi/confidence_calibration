#!/usr/bin/env python
"""decoy_settledness.py — why do good docks get LOW ipTM? Two free tests (existing structures only).

For each confident-decoy interface, compare the GOOD dock (best-DockQ structure) vs the DECOY
(max-pair_iptm structure) on:
  (1) interface pLDDT  — mean per-residue confidence (CIF B-factor) over the interface residues.
      Hypothesis: the head assigns LOWER pLDDT to the good pose (thinks the correct interface is worse).
  (2) diffusion-sample spread — for the seed that produced each pose, how much the 5 diffusion samples
      disagree at the interface (superposition-free std of cross-chain CB distances). Higher = less settled.
      Hypothesis: the good-dock seed is LESS settled (more spread) than the decoy seed.

  /home/ebrahim/esm/.venv/bin/python steer/experiments/decoy_settledness.py
"""
import csv, os, math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner

GT_DIR='/home/ebrahim/FoldBench/examples/ground_truths'
PRED='/home/ebrahim/esm/foldbench_predictions/esmfold2fast_tts'
STRUCT='results/csv_files/decoy_structural_analysis.csv'
FOLD='results/csv_files/decoy_fold_quality.csv'
DEEP='results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv'
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
        'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
        'TYR':'Y','VAL':'V'}
CBC=8.0

def parse(path):
    d=MMCIF2Dict(path); grp=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']; alt=d['_atom_site.label_alt_id']
    sid=d['_atom_site.label_seq_id']; ins=d.get('_atom_site.pdbx_PDB_ins_code',['.']*len(grp))
    b=d['_atom_site.B_iso_or_equiv']; mdl=d.get('_atom_site.pdbx_PDB_model_num',['1']*len(grp)); m0=mdl[0]
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']; res={}
    for i in range(len(grp)):
        if grp[i]!='ATOM' or mdl[i]!=m0 or comp[i] not in AA3TO1 or alt[i] not in ('.','A'): continue
        try: s=int(sid[i])
        except: continue
        rk=(ch[i],s,ins[i]); r=res.get(rk)
        if r is None: r=res[rk]={'aa':AA3TO1[comp[i]],'at':{},'b':{},'o':(s,ins[i]),'c':ch[i]}
        r['at'][atom[i]]=np.array([float(x[i]),float(y[i]),float(z[i])])
        try: r['b'][atom[i]]=float(b[i])
        except: pass
    out={}
    for rk,r in res.items():
        rep=r['at'].get('CB',r['at'].get('CA'))
        if rep is None: continue
        bf=r['b'].get('CA', next(iter(r['b'].values())) if r['b'] else float('nan'))
        out.setdefault(r['c'],[]).append({'aa':r['aa'],'rep':rep,'b':bf,'o':r['o']})
    for c in out: out[c].sort(key=lambda a:a['o'])
    return out

def seq(rl): return ''.join(r['aa'] for r in rl)
_al=PairwiseAligner(); _al.mode='global'; _al.match_score=2; _al.mismatch_score=-1
_al.open_gap_score=-5; _al.extend_gap_score=-0.5
def ident(a,b):
    if not a or not b: return 0
    al=_al.align(a,b)[0]; return sum(1 for u,v in zip(al[0],al[1]) if u==v and u!='-')/min(len(a),len(b))
def pick(pred,abs_,ags):
    abc=[c for c in pred if ident(seq(pred[c]),abs_)>=0.8]
    agc=[c for c in pred if ident(seq(pred[c]),ags)>=0.8 and c not in abc]
    if not abc or not agc: return None
    best=None
    for a in abc:
        A=np.vstack([r['rep'] for r in pred[a]])
        for g in agc:
            G=np.vstack([r['rep'] for r in pred[g]]); nc=int((cdist(A,G)<CBC).sum())
            if best is None or nc>best[0]: best=(nc,a,g)
    return best[1],best[2]
def iface_idx(ab,ag):
    A=np.vstack([r['rep'] for r in ab]); G=np.vstack([r['rep'] for r in ag]); m=cdist(A,G)<CBC
    return np.where(m.any(1))[0], np.where(m.any(0))[0]
def iface_plddt(ab,ag):
    ai,gi=iface_idx(ab,ag)
    if len(ai)<2 or len(gi)<2: return None,None
    ip=np.mean([ab[i]['b'] for i in ai]+[ag[j]['b'] for j in gi])
    ov=np.mean([r['b'] for r in ab]+[r['b'] for r in ag])
    return float(ip),float(ov)

def cif(cx,seed,samp): return os.path.join(PRED,cx,f"seed_{seed}",'predictions',f"{cx}_sample_{samp}.cif")

def spread(cx,seed,ab_chain,ag_chain,ref_mask):
    """std across the 5 diffusion samples of the distances at the REFERENCE-pose interface contacts
    (ref_mask). Chains pinned; contacts defined by the reference pose, so a jittery interface that
    only forms in some samples still registers as high spread (instead of being filtered out)."""
    mats=[]
    for s in range(5):
        p=cif(cx,seed,s)
        if not os.path.exists(p): continue
        pr=parse(p)
        if ab_chain not in pr or ag_chain not in pr: continue
        A=np.vstack([r['rep'] for r in pr[ab_chain]]); G=np.vstack([r['rep'] for r in pr[ag_chain]])
        if A.shape[0]!=ref_mask.shape[0] or G.shape[0]!=ref_mask.shape[1]: continue
        mats.append(cdist(A,G))
    if len(mats)<3 or int(ref_mask.sum())<5: return None
    return float(np.stack(mats).std(0)[ref_mask].mean())

def main():
    foldq={(r['cx'][:4],r['c1'],r['c2']):r for r in csv.DictReader(open(FOLD))}
    rows=[r for r in csv.DictReader(open(STRUCT)) if r.get('pred_ab_chain')]
    want={(r['cx'][:4],r['c1'],r['c2']) for r in rows}
    best={}
    for r in csv.DictReader(open(DEEP)):
        k=(r['assembly_id'][:4],r['chain1'],r['chain2'])
        if k not in want:
            k=(r['assembly_id'][:4],r['chain2'],r['chain1'])
            if k not in want: continue
        try: dq=float(r['dockq'])
        except: continue
        if k not in best or dq>best[k][0]: best[k]=(dq,r['seed'],r['sample'])

    out=[]; gtc={}
    for r in rows:
        cx=r['cx']; key=(cx,r['c1'],r['c2'])
        if key not in best or best[key][0]<0.4: continue
        try:
            gt=gtc.get(cx) or gtc.setdefault(cx,parse(os.path.join(GT_DIR,f'{cx}-assembly1.cif')))
            abs_=seq(gt[r['gt_ab_label']]); ags=seq(gt[r['gt_ag_label']])
            # decoy
            dp=parse(cif(cx,r['decoy_seed'],r['decoy_samp'])); dpk=pick(dp,abs_,ags)
            # good
            _,gs,gsamp=best[key]; gp=parse(cif(cx,gs,gsamp)); gpk=pick(gp,abs_,ags)
            if not dpk or not gpk: continue
            dpl,dov=iface_plddt(dp[dpk[0]],dp[dpk[1]]); gpl,gov=iface_plddt(gp[gpk[0]],gp[gpk[1]])
            if dpl is None or gpl is None: continue
            gA=np.vstack([x['rep'] for x in gp[gpk[0]]]); gG=np.vstack([x['rep'] for x in gp[gpk[1]]])
            dA=np.vstack([x['rep'] for x in dp[dpk[0]]]); dG=np.vstack([x['rep'] for x in dp[dpk[1]]])
            gsp=spread(cx,gs,gpk[0],gpk[1], cdist(gA,gG)<CBC)
            dsp=spread(cx,r['decoy_seed'],dpk[0],dpk[1], cdist(dA,dG)<CBC)
        except Exception: continue
        ag_lddt=foldq.get(key,{}).get('ag_lddt','')
        out.append(dict(key=f"{cx} {r['c1']}-{r['c2']}", good_dq=best[key][0],
                        good_iface_plddt=gpl, decoy_iface_plddt=dpl,
                        good_overall_plddt=gov, decoy_overall_plddt=dov,
                        good_spread=gsp, decoy_spread=dsp, ag_lddt=ag_lddt))

    with open('results/csv_files/decoy_settledness.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    def arr(f1,f2):
        m=[(o[f1],o[f2]) for o in out if isinstance(o[f1],float) and isinstance(o[f2],float)
           and o[f1]==o[f1] and o[f2]==o[f2]]
        return np.array([a for a,_ in m]), np.array([b for _,b in m])
    print(f"complexes compared: {len(out)}\n")
    g,d=arr('good_iface_plddt','decoy_iface_plddt')
    print("(1) INTERFACE pLDDT (head's confidence in the interface):")
    print(f"    good median={np.median(g):.1f}  decoy median={np.median(d):.1f}  "
          f"good<decoy in {int((g<d).sum())}/{len(g)}   Wilcoxon p={wilcoxon(g,d).pvalue:.4f}")
    g2,d2=arr('good_overall_plddt','decoy_overall_plddt')
    print(f"    (overall pLDDT: good {np.median(g2):.1f} vs decoy {np.median(d2):.1f})")
    gs,ds=arr('good_spread','decoy_spread')
    print(f"\n(2) DIFFUSION-SAMPLE SPREAD at interface (higher = less settled):")
    print(f"    good median={np.median(gs):.2f}A  decoy median={np.median(ds):.2f}A  "
          f"good>decoy in {int((gs>ds).sum())}/{len(gs)}   Wilcoxon p={wilcoxon(gs,ds).pvalue:.4f}")
    print("\nwrote results/csv_files/decoy_settledness.csv")

if __name__=='__main__':
    main()
