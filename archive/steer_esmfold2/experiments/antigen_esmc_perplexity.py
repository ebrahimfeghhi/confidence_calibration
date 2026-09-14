#!/usr/bin/env python
"""antigen_esmc_perplexity.py — does ESM-C's surprise at the ANTIGEN sequence predict failure?

ESM-C is a masked LM, so we compute PSEUDO-perplexity: mask each residue, read the model's
probability of the true residue given the rest, average the negative log-likelihood. Higher
pseudo-perplexity = the LM finds the antigen more surprising = (hypothesis) less well understood.

Per unique antigen (from the ground-truth antigen chain of each confident-decoy interface), then
correlate against epitope error, DockQ, ground-truth antigen lDDT, and antigen pLDDT.

  CUDA_VISIBLE_DEVICES=6 /home/ebrahim/esm/.venv/bin/python steer/experiments/antigen_esmc_perplexity.py
"""
import csv, os, math
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root
from accelerate import init_empty_weights

def load_esmc600(device):
    """Load ESM-C 600M directly from the LFS weight file (the packaged loader points at the wrong dir)."""
    root=str(data_root("esmc-600"))
    with init_empty_weights():
        m=ESMC(d_model=1152, n_heads=18, n_layers=36,
               tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False).eval()
    sd=torch.load(root+"/data/weights/esmc_600m_2024_12_v0.pth", map_location="cpu")
    m.load_state_dict(sd, assign=True)
    return m.to(device).eval()

GT_DIR='/home/ebrahim/FoldBench/examples/ground_truths'
STRUCT='results/csv_files/decoy_structural_analysis.csv'
FOLD='results/csv_files/decoy_fold_quality.csv'
PLDDT='results/csv_files/antigen_plddt_corr.csv'
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
        'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
        'TYR':'Y','VAL':'V'}

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
    tok = model.encode(ESMProtein(sequence=seq)).sequence.to(device)  # [L] incl BOS/EOS
    L = tok.shape[0]; mask_id = model.tokenizer.mask_token_id
    pos = list(range(1, L-1)); nlls=[]
    for s in range(0, len(pos), batch):
        chunk = pos[s:s+batch]
        bt = tok.unsqueeze(0).repeat(len(chunk),1).clone()
        for r,p in enumerate(chunk): bt[r,p]=mask_id
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(sequence_tokens=bt).sequence_logits           # [b,L,V]
        lp = torch.log_softmax(logits.float(), dim=-1)
        for r,p in enumerate(chunk):
            nlls.append(-lp[r,p,tok[p]].item())
    mnll=float(np.mean(nlls)); return mnll, math.exp(mnll)

def main():
    foldq={(r['cx'][:4],r['c1'],r['c2']):r for r in csv.DictReader(open(FOLD))}
    pl={r['key']:r for r in csv.DictReader(open(PLDDT))}
    rows=[]
    for r in csv.DictReader(open(STRUCT)):
        if not r.get('gt_ag_label'): continue
        key=(r['cx'][:4],r['c1'],r['c2']); kk=f"{key[0]} {key[1]}-{key[2]}"
        def num(x):
            try: return float(x)
            except: return float('nan')
        rows.append(dict(cx=r['cx'], key=kk, ag_label=r['gt_ag_label'],
                         epi=num(r.get('epitope_overlap')), good=num(r.get('good_dq')),
                         decoy=num(r.get('decoy_dq')),
                         ag_lddt=num(foldq.get(key,{}).get('ag_lddt')),
                         plddt=num(pl.get(kk,{}).get('plddt'))))
    # unique antigen sequences (dedup by complex)
    seqcache={}
    for r in rows:
        if r['cx'] in seqcache: r['seq']=seqcache[r['cx']]; continue
        try: s=chain_seq(os.path.join(GT_DIR,f"{r['cx']}-assembly1.cif"), r['ag_label'])
        except Exception: s=''
        seqcache[r['cx']]=s; r['seq']=s
    for r in rows:
        r['seq']=seqcache[r['cx']]

    device='cuda'
    print("loading ESM-C 600M ..."); model=load_esmc600(device)
    print(f"scoring {len(set(r['cx'] for r in rows))} unique antigens ...")
    ppl={}
    for cx,s in seqcache.items():
        if not s or len(s)<5: ppl[cx]=float('nan'); continue
        try: mnll,pp=pseudo_ppl(model,s,device); ppl[cx]=pp
        except Exception as e: print(f"  {cx}: {type(e).__name__} {e}"); ppl[cx]=float('nan')
        print(f"  {cx}: L={len(s)} pseudo-perplexity={ppl[cx]:.3f}")
    for r in rows: r['ppl']=ppl[r['cx']]

    def corr(f1,f2):
        m=[(r[f1],r[f2]) for r in rows if r[f1]==r[f1] and r[f2]==r[f2]]
        if len(m)<5: return None
        x=np.array([a for a,_ in m]); y=np.array([b for _,b in m])
        return len(m),pearsonr(x,y),spearmanr(x,y)
    print("\n=== antigen pseudo-perplexity correlations ===")
    for name,f in [("epitope overlap (0=wrong surface)","epi"),
                   ("best DockQ (good_dq)","good"),
                   ("ground-truth antigen lDDT","ag_lddt"),
                   ("antigen pLDDT (do the two difficulty signals agree?)","plddt")]:
        c=corr('ppl',f)
        if c: n,(pr,pp),(sr,sp)=c; print(f"perplexity vs {name}:\n"
            f"    n={n}  Pearson r={pr:+.2f} (p={pp:.3f})  Spearman ρ={sr:+.2f} (p={sp:.3f})")
    mis=[r['ppl'] for r in rows if r['ag_lddt']==r['ag_lddt'] and r['ag_lddt']<0.70 and r['ppl']==r['ppl']]
    cln=[r['ppl'] for r in rows if r['ag_lddt']==r['ag_lddt'] and r['ag_lddt']>=0.70 and r['ppl']==r['ppl']]
    if mis and cln:
        print(f"\nperplexity — misfolded antigens (n={len(mis)}) mean={np.mean(mis):.2f} vs "
              f"clean (n={len(cln)}) mean={np.mean(cln):.2f}  Mann-Whitney p={mannwhitneyu(mis,cln).pvalue:.3f}")
    with open('results/csv_files/antigen_esmc_perplexity.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cx','key','ppl','epi','good','decoy','ag_lddt','plddt'],extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print("\nwrote results/csv_files/antigen_esmc_perplexity.csv")

if __name__=='__main__':
    main()
