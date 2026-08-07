#!/usr/bin/env python
"""Did ranking donors on PARATOPE RMSD sacrifice EPITOPE accuracy?

paratopedonor picks the pose minimising paratope CA RMSD. If that pose is systematically worse at the
EPITOPE than dockqdonor's donor, the arm was never a clean test of "better binding-site fold" -- it
traded one side of the interface for the other, and a fair test needs a donor ranked on combined
paratope + epitope RMSD.

Measures both donors on both sides: CA RMSD over the native contact residues of each chain, after
whole-chain superposition (same procedure as scripts/donor_h3_contact.py).

Usage: python scripts/donor_epitope_check.py
Writes: artifacts/donor_epitope_check.json
"""
import json, os, sys, difflib, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gemmi
from decoy_set import EXP, PRED, native_chains
from fold_quality import TLC, nat_res, kind

W=f"{EXP}/artifacts/donor_epi"; os.makedirs(W, exist_ok=True)
pk=json.load(open(f"{EXP}/artifacts/paratope_donor_picks.json"))
dd=json.load(open(f"{EXP}/artifacts/dockq_arm_donors.json"))

def seq_of(rs): return "".join(TLC[r.name] for r in rs)
def ca(r):
    a=r.find_atom("CA","*"); return a.pos if a else None
def iface_idx(pdb,ch,partner):
    D=native_chains(pdb); XA,IA=D[ch]; XB,_=D[partner]
    order={v:i for i,v in enumerate(sorted(set(IA),key=lambda x:int(x)))}
    m=np.linalg.norm(XA[:,None,:]-XB[None,:,:],axis=-1)<5.0
    return {order[IA[i]] for i in np.where(m.any(1))[0]}

def contact_rmsd(pdb,ch,partner,s,k):
    nres=nat_res(pdb,ch); nseq=seq_of(nres)
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif): return None,0
    stm=gemmi.read_structure(cif); stm.setup_entities()
    best,br=None,-1
    for c in stm[0]:
        rs=[r for r in c if r.name in TLC]
        if len(rs)<4: continue
        r_=difflib.SequenceMatcher(None,seq_of(rs),nseq,autojunk=False).ratio()
        if r_>br: br,best=r_,rs
    if best is None: return None,0
    sm=difflib.SequenceMatcher(None,seq_of(best),nseq,autojunk=False)
    pairs=[(a+i,b+i) for a,b,n in sm.get_matching_blocks() for i in range(n)]
    allm=[];alln=[]
    for mi,ni in pairs:
        cm,cn=ca(best[mi]),ca(nres[ni])
        if cm and cn: allm.append(cm); alln.append(cn)
    if len(allm)<3: return None,0
    T=gemmi.superpose_positions(alln,allm).transform
    contacts=iface_idx(pdb,ch,partner); sq=[]
    for mi,ni in pairs:
        if ni not in contacts: continue
        cm,cn=ca(best[mi]),ca(nres[ni])
        if cm and cn: sq.append(gemmi.Position(T.apply(cm)).dist(cn)**2)
    return ((sum(sq)/len(sq))**0.5 if sq else None), len(contacts)

rows=[]
print(f"{'interface':11} | {'PARATOPE RMSD':>21} | {'EPITOPE RMSD':>21}")
print(f"{'':11} | {'para-donor':>10} {'dq-donor':>10} | {'para-donor':>10} {'dq-donor':>10}")
for lab in sorted(set(pk)&set(dd)):
    a=pk[lab]; b=dd[lab]; pdb=a['pdb']; ab=a['antibody_chain']; ag=a['antigen_chain']
    pa,_=contact_rmsd(pdb,ab,ag,a['seed'],a['sample'])
    ea,n_ep=contact_rmsd(pdb,ag,ab,a['seed'],a['sample'])
    pb,_=contact_rmsd(pdb,ab,ag,b['seed'],b['sample'])
    eb,_=contact_rmsd(pdb,ag,ab,b['seed'],b['sample'])
    if None in (pa,ea,pb,eb): continue
    rows.append({'interface':lab,'para_donor_paratope':round(pa,3),'dq_donor_paratope':round(pb,3),
                 'para_donor_epitope':round(ea,3),'dq_donor_epitope':round(eb,3),
                 'n_epitope_res':n_ep,
                 'same_donor':(a['seed'],a['sample'])==(b['seed'],b['sample'])})
    print(f"{lab:11} | {pa:10.2f} {pb:10.2f} | {ea:10.2f} {eb:10.2f}"
          f"{'   (same donor)' if rows[-1]['same_donor'] else ''}", flush=True)
json.dump(rows,open(f"{EXP}/artifacts/donor_epitope_check.json","w"),indent=1)
diff=[r for r in rows if not r['same_donor']]
print(f"\nMEDIANS over the {len(diff)} interfaces where the donors DIFFER:")
print(f"  paratope RMSD : para-donor {st.median(r['para_donor_paratope'] for r in diff):.2f} A"
      f"   dq-donor {st.median(r['dq_donor_paratope'] for r in diff):.2f} A")
print(f"  epitope  RMSD : para-donor {st.median(r['para_donor_epitope'] for r in diff):.2f} A"
      f"   dq-donor {st.median(r['dq_donor_epitope'] for r in diff):.2f} A")
w=sum(1 for r in diff if r['para_donor_epitope']>r['dq_donor_epitope'])
print(f"  para-donor's epitope is WORSE on {w}/{len(diff)}")
print(f"  median epitope penalty: "
      f"{st.median(r['para_donor_epitope']-r['dq_donor_epitope'] for r in diff):+.2f} A")
