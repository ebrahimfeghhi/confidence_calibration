#!/usr/bin/env python
"""decoy_complementarity.py — is the WRONG interface a better geometric/electrostatic fit than the
model's own RIGHT interface? Controls for prediction-idealization by comparing two ESMFold2 outputs.

For each clean confident-decoy interface: the DECOY = max-pair_iptm structure (from
decoy_structural_analysis.csv), the ORACLE = the model's best-DockQ structure for the same interface
(argmax dockq in the deep pool). Both are model predictions. On each, measure the antibody<->antigen
interface:
  contacts    = # interface residue pairs (CB<8A)             -> interface size
  tight_frac  = fraction of antibody interface heavy atoms with an antigen heavy atom <4A -> shape packing
  mean_gap    = mean nearest-antigen-atom distance over antibody interface atoms (lower=tighter)
  elec_comp   = mean charge product over contacting residue pairs (K/R=+1, D/E=-1); NEGATIVE = complementary
Paired decoy-vs-oracle across interfaces. If decoy packs tighter / more complementary -> better-fit lock.
  /home/ebrahim/esm/.venv/bin/python steer/experiments/decoy_complementarity.py
"""
import csv, os, math, collections
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
CH={'K':1,'R':1,'D':-1,'E':-1}
CBC=8.0; TIGHT=4.0

def parse(path,key):
    d=MMCIF2Dict(path); grp=d['_atom_site.group_PDB']
    ch=d['_atom_site.auth_asym_id'] if key=='auth' else d['_atom_site.label_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']; alt=d['_atom_site.label_alt_id']
    sid=d['_atom_site.label_seq_id']; ins=d.get('_atom_site.pdbx_PDB_ins_code',['.']*len(grp))
    mdl=d.get('_atom_site.pdbx_PDB_model_num',['1']*len(grp)); m0=mdl[0]
    x=d['_atom_site.Cartn_x']; y=d['_atom_site.Cartn_y']; z=d['_atom_site.Cartn_z']; res={}
    for i in range(len(grp)):
        if grp[i]!='ATOM' or mdl[i]!=m0 or comp[i] not in AA3TO1 or alt[i] not in ('.','A'): continue
        if atom[i].startswith('H'): continue
        try: s=int(sid[i])
        except: continue
        rk=(ch[i],s,ins[i]); r=res.get(rk)
        if r is None: r=res[rk]={'aa':AA3TO1[comp[i]],'at':{},'o':(s,ins[i]),'c':ch[i]}
        r['at'][atom[i]]=np.array([float(x[i]),float(y[i]),float(z[i])])
    out={}
    for rk,r in res.items():
        rep=r['at'].get('CB',r['at'].get('CA'))
        if rep is None: continue
        out.setdefault(r['c'],[]).append({'aa':r['aa'],'rep':rep,'heavy':np.vstack(list(r['at'].values())),'o':r['o']})
    for c in out: out[c].sort(key=lambda a:a['o'])
    return out

def seq(rl): return ''.join(r['aa'] for r in rl)
_al=PairwiseAligner(); _al.mode='global'; _al.match_score=2; _al.mismatch_score=-1
_al.open_gap_score=-5; _al.extend_gap_score=-0.5
def ident(a,b):
    if not a or not b: return 0
    al=_al.align(a,b)[0]; return sum(1 for u,v in zip(al[0],al[1]) if u==v and u!='-')/min(len(a),len(b))

def pick_chains(pred, ab_seq, ag_seq):
    abc=[c for c in pred if ident(seq(pred[c]),ab_seq)>=0.8]
    agc=[c for c in pred if ident(seq(pred[c]),ag_seq)>=0.8 and c not in abc]
    if not abc or not agc: return None
    best=None
    for a in abc:
        A=np.vstack([r['rep'] for r in pred[a]])
        for g in agc:
            G=np.vstack([r['rep'] for r in pred[g]])
            nc=int((cdist(A,G)<CBC).sum())
            if best is None or nc>best[0]: best=(nc,a,g)
    return best[1],best[2]

def metrics(ab,ag):
    A=np.vstack([r['rep'] for r in ab]); G=np.vstack([r['rep'] for r in ag])
    D=cdist(A,G); pair=D<CBC
    ai=np.where(pair.any(axis=1))[0]; gi=np.where(pair.any(axis=0))[0]
    if len(ai)<2 or len(gi)<2: return None
    Ah=np.vstack([ab[i]['heavy'] for i in ai]); Gh=np.vstack([ag[i]['heavy'] for i in gi])
    Dh=cdist(Ah,Gh); nn=Dh.min(axis=1)
    tight=float((nn<TIGHT).mean()); gap=float(nn.mean())
    # electrostatic complementarity over contacting residue pairs
    prods=[]
    for i in ai:
        for j in gi:
            if pair[i,j]:
                prods.append(CH.get(ab[i]['aa'],0)*CH.get(ag[j]['aa'],0))
    elec=float(np.mean(prods)) if prods else 0.0
    return dict(contacts=int(pair.sum()), tight=tight, gap=gap, elec=elec)

def main():
    bad=set()
    for r in csv.DictReader(open(FOLD)):
        try: ag=float(r['ag_lddt'])
        except: ag=float('nan')
        if not(ag==ag) or ag<0.70: bad.add((r['cx'][:4],r['c1'],r['c2']))
    rows=[r for r in csv.DictReader(open(STRUCT))
          if (r['cx'][:4],r['c1'],r['c2']) not in bad and r.get('pred_ab_chain')]
    want={(r['cx'][:4],r['c1'],r['c2']) for r in rows}
    # oracle (max-dockq seed,sample) per interface from deep pool
    best={}
    for r in csv.DictReader(open(DEEP)):
        k=(r['assembly_id'][:4],r['chain1'],r['chain2'])
        if k not in want:
            k=(r['assembly_id'][:4],r['chain2'],r['chain1'])
            if k not in want: continue
        try: dq=float(r['dockq'])
        except: continue
        if k not in best or dq>best[k][0]: best[k]=(dq,r['seed'],r['sample'])

    keys=['contacts','tight','gap','elec']; dec={k:[] for k in keys}; orc={k:[] for k in keys}
    used=0; gtc={}
    for r in rows:
        cx=r['cx']; key=(cx,r['c1'],r['c2'])
        if key not in best or best[key][0]<0.4: continue          # need a genuine good dock
        try:
            gt=gtc.get(cx) or gtc.setdefault(cx,parse(os.path.join(GT_DIR,f'{cx}-assembly1.cif'),'label'))
            abs_=seq(gt[r['gt_ab_label']]); ags=seq(gt[r['gt_ag_label']])
            dcif=os.path.join(PRED,cx,f"seed_{r['decoy_seed']}",'predictions',f"{cx}_sample_{r['decoy_samp']}.cif")
            _,os_,ss=best[key]; ocif=os.path.join(PRED,cx,f"seed_{os_}",'predictions',f"{cx}_sample_{ss}.cif")
            dp=parse(dcif,'auth'); op=parse(ocif,'auth')
            dch=pick_chains(dp,abs_,ags); och=pick_chains(op,abs_,ags)
            if not dch or not och: continue
            md=metrics(dp[dch[0]],dp[dch[1]]); mo=metrics(op[och[0]],op[och[1]])
            if md is None or mo is None: continue
        except Exception: continue
        for k in keys: dec[k].append(md[k]); orc[k].append(mo[k])
        used+=1
    print(f"interfaces compared (decoy vs model's own good dock): {used}\n")
    lab={'contacts':'interface contacts','tight':'tight-packed atom frac (<4A)',
         'gap':'mean interface gap (A, lower=tighter)','elec':'electrostatic complementarity (neg=complementary)'}
    print(f"{'metric':>34} {'decoy':>8} {'oracle':>8} {'wilcoxon p':>11}  read")
    for k in keys:
        d=np.array(dec[k],float); o=np.array(orc[k],float)
        try: p=wilcoxon(d,o).pvalue
        except: p=float('nan')
        better = "decoy TIGHTER/bigger" if (k!='gap' and d.mean()>o.mean()) or (k=='gap' and d.mean()<o.mean()) else "oracle better/≈"
        print(f"{lab[k]:>34} {d.mean():>8.2f} {o.mean():>8.2f} {p:>11.3f}  {better}")
    print("\n(decoy = confident wrong dock; oracle = model's own best-DockQ dock; both ESMFold2 outputs)")

if __name__=='__main__':
    main()
