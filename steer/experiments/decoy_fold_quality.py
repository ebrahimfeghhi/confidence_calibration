#!/usr/bin/env python
"""decoy_fold_quality.py — is each confident decoy a mis-DOCK or a mis-FOLD?

For every confident-decoy interface (decoy_structural_analysis.csv pins the exact decoy structure +
chain IDs), compute per-chain CA-lDDT vs ground truth for the antibody chain and the antigen chain.
CA-lDDT is superposition-free and INTRA-chain only, so it measures fold quality independent of how
the two chains are docked. High lDDT on both chains => the low DockQ is a pure docking error;
low lDDT => that chain actually misfolded.

  /home/ebrahim/esm/.venv/bin/python steer/experiments/decoy_fold_quality.py
Writes results/csv_files/decoy_fold_quality.csv + prints a summary.
"""
import os, csv
import numpy as np
from scipy.spatial.distance import cdist
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner

GT_DIR    = '/home/ebrahim/FoldBench/examples/ground_truths'
PRED_ROOT = '/home/ebrahim/esm/foldbench_predictions/esmfold2fast_tts'
IN_CSV    = 'results/csv_files/decoy_structural_analysis.csv'
OUT_CSV   = 'results/csv_files/decoy_fold_quality.csv'
AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
          'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
          'TYR':'Y','VAL':'V'}

def parse_cif(path, chain_key):
    d = MMCIF2Dict(path)
    grp=d['_atom_site.group_PDB']
    ch=d['_atom_site.auth_asym_id'] if chain_key=='auth' else d['_atom_site.label_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']; alt=d['_atom_site.label_alt_id']
    seqid=d['_atom_site.label_seq_id']; ins=d.get('_atom_site.pdbx_PDB_ins_code',['.']*len(grp))
    model=d.get('_atom_site.pdbx_PDB_model_num',['1']*len(grp))
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']; m0=model[0]
    res={}
    for i in range(len(grp)):
        if grp[i]!='ATOM' or model[i]!=m0 or comp[i] not in AA3TO1 or alt[i] not in ('.','A'): continue
        try: sid=int(seqid[i])
        except ValueError: continue
        rk=(ch[i],sid,ins[i]); r=res.get(rk)
        if r is None: r=res[rk]={'resname':comp[i],'atoms':{},'order':(sid,ins[i]),'chain':ch[i]}
        if atom[i] not in r['atoms']: r['atoms'][atom[i]]=np.array([float(x[i]),float(y[i]),float(z[i])])
    chains={}
    for rk,r in res.items():
        ca=r['atoms'].get('CA')
        chains.setdefault(r['chain'],[]).append({'aa':AA3TO1[r['resname']],'ca':ca,'order':r['order']})
    for c in chains: chains[c].sort(key=lambda a:a['order'])
    return chains

def seq(rl): return ''.join(r['aa'] for r in rl)

_al=PairwiseAligner(); _al.mode='global'; _al.match_score=2; _al.mismatch_score=-1
_al.open_gap_score=-5; _al.extend_gap_score=-0.5
def map_indices(src,dst):
    aln=_al.align(src,dst)[0]; ta,tb=aln[0],aln[1]; mp={}; si=di=0
    for ca,cb in zip(ta,tb):
        if ca!='-' and cb!='-': mp[si]=di
        if ca!='-': si+=1
        if cb!='-': di+=1
    return mp

def ca_lddt(pred_rl, gt_rl, R0=15.0, thr=(0.5,1,2,4)):
    """intra-chain CA-lDDT of pred vs gt over aligned residues with CA on both sides."""
    mp=map_indices(seq(pred_rl), seq(gt_rl))
    P=[]; G=[]
    for i,j in mp.items():
        pca=pred_rl[i]['ca']; gca=gt_rl[j]['ca']
        if pca is not None and gca is not None: P.append(pca); G.append(gca)
    if len(P)<4: return float('nan'), len(P)
    P=np.vstack(P); G=np.vstack(G)
    Dr=cdist(G,G); Dm=cdist(P,P)
    mask=(Dr<R0)&(Dr>0); npair=mask.sum()
    if npair==0: return float('nan'), len(P)
    diff=np.abs(Dr-Dm)
    s=sum(((diff<t)&mask).sum()/npair for t in thr)/len(thr)
    return float(s), len(P)

def band(o):
    try: o=float(o)
    except: return "n/a"
    if o!=o: return "n/a"
    if o<0.1: return "wrong surface"
    if o<0.5: return "wrong register"
    return "bad pose"

def main():
    rows=list(csv.DictReader(open(IN_CSV)))
    out=[]; gt_cache={}
    for r in rows:
        cx=r['cx']; rec=dict(cx=cx, c1=r['c1'], c2=r['c2'], decoy_dq=r['decoy_dq'],
                             epitope_overlap=r['epitope_overlap'], ag_band=band(r['epitope_overlap']))
        if not r.get('pred_ab_chain') or not r.get('pred_ag_chain'):
            rec['note']='no_pred_chain'; out.append(rec); continue
        try:
            gtp=os.path.join(GT_DIR,f'{cx}-assembly1.cif')
            gt=gt_cache.get(cx) or gt_cache.setdefault(cx, parse_cif(gtp,'label'))
            pp=os.path.join(PRED_ROOT,cx,f"seed_{r['decoy_seed']}",'predictions',f"{cx}_sample_{r['decoy_samp']}.cif")
            pred=parse_cif(pp,'auth')
            ab_l,ag_l=r['gt_ab_label'],r['gt_ag_label']; ab_p,ag_p=r['pred_ab_chain'],r['pred_ag_chain']
            ab_lddt,ab_n=ca_lddt(pred[ab_p],gt[ab_l]) if ab_l in gt and ab_p in pred else (float('nan'),0)
            ag_lddt,ag_n=ca_lddt(pred[ag_p],gt[ag_l]) if ag_l in gt and ag_p in pred else (float('nan'),0)
        except Exception as e:
            rec['note']=f'err:{type(e).__name__}'; out.append(rec); continue
        rec.update(ab_lddt=round(ab_lddt,3), ag_lddt=round(ag_lddt,3), ab_n=ab_n, ag_n=ag_n, note='')
        out.append(rec)

    os.makedirs(os.path.dirname(OUT_CSV),exist_ok=True)
    cols=['cx','c1','c2','decoy_dq','epitope_overlap','ag_band','ab_lddt','ag_lddt','ab_n','ag_n','note']
    with open(OUT_CSV,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(out)

    # ---- summary ----
    good=[r for r in out if isinstance(r.get('ab_lddt'),float) and r['ab_lddt']==r['ab_lddt']
          and isinstance(r.get('ag_lddt'),float) and r['ag_lddt']==r['ag_lddt']]
    def verdict(r):
        THR=0.70
        ab,ag=r['ab_lddt'],r['ag_lddt']
        if min(ab,ag)>=THR: return "both folded -> pure DOCK error"
        if ab<THR and ag>=THR: return "ANTIBODY misfolded"
        if ag<THR and ab>=THR: return "ANTIGEN misfolded"
        return "both chains poor"
    print(f"{'interface':>12} {'decoyDQ':>7} {'ab_lddt':>7} {'ag_lddt':>7}  {'ag_band':<14} verdict")
    import collections
    vc=collections.Counter()
    for r in sorted(good,key=lambda r:min(r['ab_lddt'],r['ag_lddt'])):
        v=verdict(r); vc[v]+=1
        print(f"{r['cx']+' '+r['c1']+'-'+r['c2']:>12} {float(r['decoy_dq']):7.2f} "
              f"{r['ab_lddt']:7.2f} {r['ag_lddt']:7.2f}  {r['ag_band']:<14} {v}")
    n=len(good)
    print(f"\nscored {n} interfaces (of {len(out)}); CA-lDDT threshold 0.70 = 'correctly folded backbone'")
    for v,c in vc.most_common(): print(f"  {c:>2} ({100*c/n:.0f}%)  {v}")
    ab=[r['ab_lddt'] for r in good]; ag=[r['ag_lddt'] for r in good]
    print(f"\nmean CA-lDDT: antibody={np.mean(ab):.2f}  antigen={np.mean(ag):.2f}")
    print(f"wrote {OUT_CSV}")

if __name__=='__main__':
    main()
