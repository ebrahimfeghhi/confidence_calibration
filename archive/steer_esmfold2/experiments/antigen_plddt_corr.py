#!/usr/bin/env python
"""antigen_plddt_corr.py — does the model's own confidence in the ANTIGEN fold predict failure?

pLDDT is already stored in the predicted CIFs (B-factor column). For each confident-decoy interface,
take the mean pLDDT of the antigen chain (from the decoy structure) and correlate against:
  ag_lddt (ground-truth antigen fold accuracy)  -> does self-confidence track real accuracy?
  epitope_overlap (0=wrong surface)              -> does low antigen confidence predict wrong epitope?
  good_dq (best reachable DockQ)                 -> does it predict overall dock quality?
Also: is antigen pLDDT lower for the misfolded-antigen cases (ag_lddt<0.70)?

  /home/ebrahim/esm/.venv/bin/python steer/experiments/antigen_plddt_corr.py
"""
import csv, os
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

PRED='/home/ebrahim/esm/foldbench_predictions/esmfold2fast_tts'
STRUCT='results/csv_files/decoy_structural_analysis.csv'
FOLD='results/csv_files/decoy_fold_quality.csv'
STD3={'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO',
      'SER','THR','TRP','TYR','VAL'}

def chain_plddt(path, chain):
    """mean CA pLDDT (B-factor) over residues of `chain` (auth id)."""
    d=MMCIF2Dict(path); grp=d['_atom_site.group_PDB']; ch=d['_atom_site.auth_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']
    b=d['_atom_site.B_iso_or_equiv']; mdl=d.get('_atom_site.pdbx_PDB_model_num',['1']*len(grp)); m0=mdl[0]
    vals=[]
    for i in range(len(grp)):
        if grp[i]!='ATOM' or mdl[i]!=m0 or ch[i]!=chain or atom[i]!='CA' or comp[i] not in STD3: continue
        try: vals.append(float(b[i]))
        except: pass
    return float(np.mean(vals)) if vals else float('nan')

def main():
    foldq={(r['cx'][:4],r['c1'],r['c2']):r for r in csv.DictReader(open(FOLD))}
    rows=[]
    for r in csv.DictReader(open(STRUCT)):
        key=(r['cx'][:4],r['c1'],r['c2'])
        if not r.get('pred_ag_chain'): continue
        cif=os.path.join(PRED,r['cx'],f"seed_{r['decoy_seed']}",'predictions',f"{r['cx']}_sample_{r['decoy_samp']}.cif")
        if not os.path.exists(cif): continue
        try: pl=chain_plddt(cif, r['pred_ag_chain'])
        except Exception: continue
        fq=foldq.get(key,{})
        def num(x):
            try: return float(x)
            except: return float('nan')
        rows.append(dict(key=f"{key[0]} {key[1]}-{key[2]}", plddt=pl,
                         ag_lddt=num(fq.get('ag_lddt')), epi=num(r.get('epitope_overlap')),
                         good=num(r.get('good_dq')), decoy=num(r.get('decoy_dq'))))
    def corr(xs,ys):
        m=[(x,y) for x,y in zip(xs,ys) if x==x and y==y]
        if len(m)<5: return None
        x=np.array([a for a,_ in m]); y=np.array([b for _,b in m])
        return len(m), pearsonr(x,y), spearmanr(x,y)

    pl=[r['plddt'] for r in rows]
    print(f"antigen pLDDT: n={len(rows)} mean={np.nanmean(pl):.1f} range={np.nanmin(pl):.0f}-{np.nanmax(pl):.0f}\n")
    for name,field in [("ground-truth antigen lDDT (validation)","ag_lddt"),
                       ("epitope overlap (0=wrong surface)","epi"),
                       ("best reachable DockQ (good_dq)","good")]:
        c=corr(pl,[r[field] for r in rows])
        if c: n,(pr,pp),(sr,sp)=c; print(f"antigen pLDDT vs {name}:\n"
              f"    n={n}  Pearson r={pr:+.2f} (p={pp:.3f})  Spearman ρ={sr:+.2f} (p={sp:.3f})")
    # misfold vs clean
    mis=[r['plddt'] for r in rows if r['ag_lddt']==r['ag_lddt'] and r['ag_lddt']<0.70]
    cln=[r['plddt'] for r in rows if r['ag_lddt']==r['ag_lddt'] and r['ag_lddt']>=0.70]
    if mis and cln:
        u=mannwhitneyu(mis,cln)
        print(f"\nantigen pLDDT — misfolded antigens (n={len(mis)}) mean={np.mean(mis):.1f}  "
              f"vs clean (n={len(cln)}) mean={np.mean(cln):.1f}   Mann-Whitney p={u.pvalue:.3f}")
    os.makedirs('results/csv_files',exist_ok=True)
    with open('results/csv_files/antigen_plddt_corr.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['key','plddt','ag_lddt','epi','good','decoy']); w.writeheader(); w.writerows(rows)
    print("\nwrote results/csv_files/antigen_plddt_corr.csv")

if __name__=='__main__':
    main()
