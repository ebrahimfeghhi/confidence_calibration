#!/usr/bin/env python
"""antigen_perplexity_benchmark.py — do CONFIDENTLY-WRONG-epitope complexes have higher-perplexity
antigens than complexes the model gets right?

Group A (confidently wrong epitope): a good dock exists (best_dockq>=0.5) but ipTM selects a decoy
   (iptm_sel_dockq<0.23).  Group B (selection ok): ipTM selects a good dock (iptm_sel_dockq>=0.5).
Antigen perplexity (ESM-C 600M, masked pseudo-perplexity) from the ground-truth antigen chain of
every benchmark complex. Compare A vs B (Mann-Whitney), overall and excluding known misfolded antigens.

  CUDA_VISIBLE_DEVICES=6 /home/ebrahim/esm/.venv/bin/python steer/experiments/antigen_perplexity_benchmark.py
"""
import csv, os, math
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root
from accelerate import init_empty_weights

GT_DIR='/home/ebrahim/FoldBench/examples/ground_truths'
ROLES='/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv'
GAP='results/csv_files/iptm_selection_gap_baseline.csv'
FOLD='results/csv_files/decoy_fold_quality.csv'
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
        'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
        'TYR':'Y','VAL':'V'}

def load_esmc600(device):
    root=str(data_root("esmc-600"))
    with init_empty_weights():
        m=ESMC(d_model=1152,n_heads=18,n_layers=36,tokenizer=get_esmc_model_tokenizers(),use_flash_attn=False).eval()
    sd=torch.load(root+"/data/weights/esmc_600m_2024_12_v0.pth",map_location="cpu")
    m.load_state_dict(sd,assign=True); return m.to(device).eval()

def chain_seq(path, chain_label):
    d=MMCIF2Dict(path); grp=d['_atom_site.group_PDB']; ch=d['_atom_site.label_asym_id']
    comp=d['_atom_site.label_comp_id']; atom=d['_atom_site.label_atom_id']
    sid=d['_atom_site.label_seq_id']; ins=d.get('_atom_site.pdbx_PDB_ins_code',['.']*len(grp))
    mdl=d.get('_atom_site.pdbx_PDB_model_num',['1']*len(grp)); m0=mdl[0]; seen={}
    for i in range(len(grp)):
        if grp[i]!='ATOM' or mdl[i]!=m0 or ch[i]!=chain_label or atom[i]!='CA' or comp[i] not in AA3TO1: continue
        try: k=(int(sid[i]),ins[i])
        except: continue
        if k not in seen: seen[k]=AA3TO1[comp[i]]
    return ''.join(seen[k] for k in sorted(seen))

@torch.no_grad()
def pseudo_ppl(model, seq, device, batch=24):
    tok=model.encode(ESMProtein(sequence=seq)).sequence.to(device); L=tok.shape[0]
    mask=model.tokenizer.mask_token_id; pos=list(range(1,L-1)); nlls=[]
    for s in range(0,len(pos),batch):
        ch=pos[s:s+batch]; bt=tok.unsqueeze(0).repeat(len(ch),1).clone()
        for r,p in enumerate(ch): bt[r,p]=mask
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            logits=model(sequence_tokens=bt).sequence_logits
        lp=torch.log_softmax(logits.float(),dim=-1)
        for r,p in enumerate(ch): nlls.append(-lp[r,p,tok[p]].item())
    return math.exp(float(np.mean(nlls)))

def main():
    # antigen chain per complex (first interface row)
    agchain={}
    for r in csv.DictReader(open(ROLES)):
        cx=r['pdb_id'][:4]; agchain.setdefault(cx, r['antigen_chain'])
    # selection outcome per complex
    sel={}
    for r in csv.DictReader(open(GAP)):
        sel[r['complex'][:4]]=(float(r['best_dockq']), float(r['iptm_sel_dockq']))
    # known misfolded antigens (ag_lddt<0.70) from the confident-decoy fold check
    misfold=set(r['cx'][:4] for r in csv.DictReader(open(FOLD)) if _f(r.get('ag_lddt'))<0.70)

    device='cuda'; print("loading ESM-C 600M ..."); model=load_esmc600(device)
    rows=[]
    for cx,ag in sorted(agchain.items()):
        if cx not in sel: continue
        gt=os.path.join(GT_DIR,f'{cx}-assembly1.cif')
        if not os.path.exists(gt): continue
        try: s=chain_seq(gt, ag)
        except Exception: s=''
        if not s or len(s)<5: continue
        try: pp=pseudo_ppl(model,s,device)
        except Exception as e: print(f"  {cx}: {type(e).__name__}"); continue
        bd,sd=sel[cx]
        grp = 'A_confwrong' if (bd>=0.5 and sd<0.23) else ('B_selok' if sd>=0.5 else 'grey')
        rows.append(dict(cx=cx, ppl=pp, best_dockq=bd, sel_dockq=sd, group=grp,
                         misfold=int(cx in misfold), L=len(s)))
        print(f"  {cx}: L={len(s)} ppl={pp:6.2f} group={grp} misfold={int(cx in misfold)}")

    with open('results/csv_files/antigen_perplexity_benchmark.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cx','ppl','best_dockq','sel_dockq','group','misfold','L'])
        w.writeheader(); w.writerows(rows)

    def ppls(grp, exclude_misfold=False):
        return [r['ppl'] for r in rows if r['group']==grp and not (exclude_misfold and r['misfold'])]
    A=ppls('A_confwrong'); B=ppls('B_selok')
    print(f"\n=== antigen perplexity: confidently-wrong (A) vs selection-ok (B) ===")
    print(f"A (n={len(A)}) median={np.median(A):.2f} mean={np.mean(A):.2f}")
    print(f"B (n={len(B)}) median={np.median(B):.2f} mean={np.mean(B):.2f}")
    print(f"Mann-Whitney (A>B?) p={mannwhitneyu(A,B,alternative='greater').pvalue:.4f}")
    Ac=ppls('A_confwrong',exclude_misfold=True)
    print(f"\n-- excluding known misfolded antigens (the interesting, fold-independent test) --")
    print(f"A_wellfolded (n={len(Ac)}) median={np.median(Ac):.2f} mean={np.mean(Ac):.2f}  vs B median={np.median(B):.2f}")
    print(f"Mann-Whitney (A_wellfolded>B?) p={mannwhitneyu(Ac,B,alternative='greater').pvalue:.4f}")
    print("\nwrote results/csv_files/antigen_perplexity_benchmark.csv")

def _f(x):
    try: return float(x)
    except: return float('nan')

if __name__=='__main__':
    main()
