#!/usr/bin/env python
"""decoy_epitope_bias.py — is the WRONG epitope a consistent kind of surface?

For each clean confident-decoy interface (antigen well-folded, ag_lddt>=0.70), compare three residue
groups ON THE GROUND-TRUTH antigen structure:
  D = decoy epitope   (antigen residues the model docks the antibody onto, mapped GT<-pred)
  G = true epitope    (antigen residues the real antibody binds)
  background = whole antigen
Surface properties per antigen residue: hydrophobicity (Kyte-Doolittle), exposure (= -burial, where
burial = #CB neighbors within 10A; more neighbors = more buried), charged (DEKR), aromatic (FWY),
polar (STNQHYWC+charged). Each property is z-scored WITHIN each antigen, so a group mean-z > 0 means
"enriched vs this antigen's own average". Aggregate mean-z across interfaces (test vs 0) and D-vs-G paired.

  /home/ebrahim/esm/.venv/bin/python steer/experiments/decoy_epitope_bias.py
"""
import csv, os, math, collections
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, ttest_1samp
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner

GT_DIR='/home/ebrahim/FoldBench/examples/ground_truths'
PRED_ROOT='/home/ebrahim/esm/foldbench_predictions/esmfold2fast_tts'
STRUCT='results/csv_files/decoy_structural_analysis.csv'
FOLD='results/csv_files/decoy_fold_quality.csv'
CONTACT=8.0; RNBR=10.0
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
        'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
        'TYR':'Y','VAL':'V'}
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,
    'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGED=set('DEKR'); AROM=set('FWY'); POLAR=set('STNQHYWCDEKR')

def parse_cif(path, key):
    d=MMCIF2Dict(path); grp=d['_atom_site.group_PDB']
    ch=d['_atom_site.auth_asym_id'] if key=='auth' else d['_atom_site.label_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']; alt=d['_atom_site.label_alt_id']
    seqid=d['_atom_site.label_seq_id']; ins=d.get('_atom_site.pdbx_PDB_ins_code',['.']*len(grp))
    mdl=d.get('_atom_site.pdbx_PDB_model_num',['1']*len(grp)); m0=mdl[0]
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']; res={}
    for i in range(len(grp)):
        if grp[i]!='ATOM' or mdl[i]!=m0 or comp[i] not in AA3TO1 or alt[i] not in ('.','A'): continue
        try: sid=int(seqid[i])
        except: continue
        rk=(ch[i],sid,ins[i]); r=res.get(rk)
        if r is None: r=res[rk]={'aa':AA3TO1[comp[i]],'atoms':{},'order':(sid,ins[i]),'ch':ch[i]}
        if atom[i] not in r['atoms']: r['atoms'][atom[i]]=np.array([float(x[i]),float(y[i]),float(z[i])])
    ch2={}
    for rk,r in res.items():
        rep=r['atoms'].get('CB') if r['atoms'].get('CB') is not None else r['atoms'].get('CA')
        if rep is None: continue
        ch2.setdefault(r['ch'],[]).append({'aa':r['aa'],'rep':rep,'order':r['order']})
    for c in ch2: ch2[c].sort(key=lambda a:a['order'])
    return ch2

def seq(rl): return ''.join(r['aa'] for r in rl)
_al=PairwiseAligner(); _al.mode='global'; _al.match_score=2; _al.mismatch_score=-1
_al.open_gap_score=-5; _al.extend_gap_score=-0.5
def map_idx(src,dst):
    a=_al.align(src,dst)[0]; ta,tb=a[0],a[1]; mp={}; si=di=0
    for ca,cb in zip(ta,tb):
        if ca!='-' and cb!='-': mp[si]=di
        if ca!='-': si+=1
        if cb!='-': di+=1
    return mp
def epitope(ag,ab):
    A=np.vstack([r['rep'] for r in ag]); B=np.vstack([r['rep'] for r in ab])
    return set(np.where((cdist(A,B)<CONTACT).any(axis=1))[0].tolist())

def props(ag):
    X=np.vstack([r['rep'] for r in ag]); n=len(ag)
    D=cdist(X,X); burial=((D<RNBR)&(D>0)).sum(axis=1).astype(float)
    aa=[r['aa'] for r in ag]
    return {
      'hydrophobic':np.array([KD[a] for a in aa]),
      'exposed':-burial,                                   # higher = more exposed/protruding
      'charged':np.array([1.0 if a in CHARGED else 0.0 for a in aa]),
      'aromatic':np.array([1.0 if a in AROM else 0.0 for a in aa]),
      'polar':np.array([1.0 if a in POLAR else 0.0 for a in aa]),
    }

def zscore(v):
    s=v.std(); return (v-v.mean())/s if s>1e-9 else v*0.0

def main():
    bad=set()
    for r in csv.DictReader(open(FOLD)):
        try: ag=float(r['ag_lddt'])
        except: ag=float('nan')
        if not (ag==ag) or ag<0.70: bad.add((r['cx'][:4],r['c1'],r['c2']))
    rows=[r for r in csv.DictReader(open(STRUCT))
          if (r['cx'][:4],r['c1'],r['c2']) not in bad and r.get('pred_ab_chain') and r.get('pred_ag_chain')]
    PROPS=['hydrophobic','exposed','charged','aromatic','polar']
    zD={p:[] for p in PROPS}; zG={p:[] for p in PROPS}; used=0; gtcache={}
    for r in rows:
        cx=r['cx']
        try:
            gt=gtcache.get(cx) or gtcache.setdefault(cx,parse_cif(os.path.join(GT_DIR,f'{cx}-assembly1.cif'),'label'))
            pp=os.path.join(PRED_ROOT,cx,f"seed_{r['decoy_seed']}",'predictions',f"{cx}_sample_{r['decoy_samp']}.cif")
            pred=parse_cif(pp,'auth')
            ag_gt=gt[r['gt_ag_label']]; ab_gt=gt[r['gt_ab_label']]
            ag_pd=pred[r['pred_ag_chain']]; ab_pd=pred[r['pred_ab_chain']]
        except Exception: continue
        Gset=epitope(ag_gt,ab_gt)
        Dpred=epitope(ag_pd,ab_pd); mp=map_idx(seq(ag_pd),seq(ag_gt))
        Dset=set(mp[i] for i in Dpred if i in mp)
        if len(Gset)<3 or len(Dset)<3: continue
        P=props(ag_gt); used+=1
        for p in PROPS:
            z=zscore(P[p])
            zD[p].append(float(np.mean([z[i] for i in Dset])))
            zG[p].append(float(np.mean([z[i] for i in Gset])))
    print(f"interfaces used: {used}\n")
    print(f"{'property':>12} {'decoy-epi z':>12} {'true-epi z':>11} {'D vs 0 p':>9} {'D vs G p':>9}  read")
    summ={}
    for p in PROPS:
        d=np.array(zD[p]); g=np.array(zG[p])
        pd0=ttest_1samp(d,0).pvalue
        try: pdg=wilcoxon(d,g).pvalue
        except Exception: pdg=float('nan')
        arrow = "enriched" if d.mean()>0.15 else ("depleted" if d.mean()<-0.15 else "~neutral")
        vsg = "more than true" if d.mean()-g.mean()>0.1 else ("less than true" if d.mean()-g.mean()<-0.1 else "≈true")
        print(f"{p:>12} {d.mean():>+12.2f} {g.mean():>+11.2f} {pd0:>9.3f} {pdg:>9.3f}  decoy-epi {arrow}, {vsg}")
        summ[p]=(d.mean(),g.mean(),d.std()/max(1,len(d))**0.5,g.std()/max(1,len(g))**0.5)
    import json
    json.dump({p:[float(v) for v in summ[p]] for p in PROPS}, open('/tmp/epi_bias_summ.json','w'))
    print("\n(z>0 = enriched vs that antigen's own average surface; exposed = -burial)")

if __name__=='__main__':
    main()
