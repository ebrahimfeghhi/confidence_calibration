#!/usr/bin/env python
"""dockqdonor's donor vs the best-DockQ fold the LEADER shortlist offers -- DockQ and fold quality.

dockqdonor picks the best-DockQ pose out of all 250 free-run poses (an oracle choice). leader20 can
only use one of the 20 NMS-selected leaders. This measures what that restriction costs, on both axes:

    DockQ of the donor        how well that pose docked
    interface lDDT per chain  how well its binding site is folded (paratope and epitope separately)

On 8che_BE the deciding variable was the EPITOPE fold: identical antibody folds (0.897-0.918) across
all candidates, but chain E at 0.665 for dockqdonor's donor versus 0.392-0.625 for the leaders -- and
that is what separated 65% acceptable poses at ipTM 0.702 from 18% at 0.393.

Usage: python scripts/donor_vs_bestleader.py
Writes: artifacts/donor_vs_bestleader.json
"""
import json, os, sys, subprocess, difflib, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gemmi
from decoy_set import OST, ENV, PRED, EXP, native_chains, THRESH
from fold_quality import TLC, nat_res, nat_pdb

W=f"{EXP}/artifacts/donor_cmp"; os.makedirs(W,exist_ok=True)
don=json.load(open(f"{EXP}/artifacts/dockq_arm_donors.json"))
idx=json.load(open(f"{EXP}/artifacts/leader20_index.json"))
dq=json.load(open(f"{EXP}/artifacts/dockqdonor_scores.json"))
ld=json.load(open(f"{EXP}/artifacts/leader20_scores.json"))

def iface_idx(pdb,ch,partner):
    D=native_chains(pdb); XA,IA=D[ch]; XB,_=D[partner]
    order={v:i for i,v in enumerate(sorted(set(IA),key=lambda x:int(x)))}
    m=np.linalg.norm(XA[:,None,:]-XB[None,:,:],axis=-1)<5.0
    return {order[IA[i]] for i in np.where(m.any(1))[0]}

def iface_lddt(pdb,ch,partner,s,k,tag):
    seq="".join(TLC[r.name] for r in nat_res(pdb,ch)); ref=nat_pdb(pdb,ch)
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif): return None
    st_=gemmi.read_structure(cif); st_.setup_entities()
    best,br=None,-1
    for c in st_[0]:
        rs=[r for r in c if r.name in TLC]
        if not rs: continue
        r_=difflib.SequenceMatcher(None,"".join(TLC[x.name] for x in rs),seq,autojunk=False).ratio()
        if r_>br: br,best=r_,rs
    if best is None: return None
    mp=f"{W}/m_{tag}_{ch}.pdb"
    ns=gemmi.Structure(); ns.add_model(gemmi.Model("1")); cc=gemmi.Chain("A")
    for r in best: cc.add_residue(r)
    ns[0].add_chain(cc); ns.setup_entities(); ns.write_pdb(mp)
    o=f"{W}/ld_{tag}_{ch}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset","-c","0-63",OST,"compare-structures","-m",mp,"-r",ref,"-o",o,
                        "--fault-tolerant","--min-pep-length","4","--lddt","--local-lddt"],
                       env=ENV,capture_output=True,timeout=600)
    try: d=json.load(open(o))
    except Exception: return None
    ll=d.get("local_lddt") or {}; aln=(d.get("aln") or [None])[0]
    if not aln: return None
    p=aln.split("\n"); ri=mi=0; per={}
    for a,b in zip(p[1],p[3]):
        if a!="-" and b!="-":
            v=ll.get(f"A.{mi+1}.")
            if v is not None: per[ri]=v
        if a!="-": ri+=1
        if b!="-": mi+=1
    ii=iface_idx(pdb,ch,partner); vals=[per[i] for i in ii if i in per]
    return st.mean(vals) if vals else None

rows=[]
print(f"{'interface':11} | {'dockqdonor donor':>24} | {'best-DockQ leader':>26} | {'loss':>12}")
print(f"{'':11} | {'DockQ':>6} {'ifaceL min':>10} {'':6} | {'rank':>4} {'DockQ':>6} {'ifaceL min':>10} {'':4} | "
      f"{'DockQ':>6} {'lDDT':>5}")
# 7zau_AB has no dockqdonor score (its run was stopped), so intersect all three sources
for lab in sorted(set(don)&set(idx)&set(dq)):
    d=don[lab]; pdb=d['pdb']; c1,c2=d['interface']
    L=[x for x in idx[lab]['leaders']]
    bl=max(L,key=lambda x:x['dockq_gt']) if L else None
    if not bl: continue
    def mm(s,k,tag):
        a=iface_lddt(pdb,c1,c2,s,k,tag); b=iface_lddt(pdb,c2,c1,s,k,tag)
        return None if (a is None or b is None) else min(a,b)
    dm=mm(d['seed'],d['sample'],f"{lab}_dq")
    lm=mm(bl['seed'],bl['sample'],f"{lab}_L{bl['rank']}")
    if dm is None or lm is None: continue
    rows.append({"interface":lab,"dq_donor_dockq":d['donor_dockq'],"dq_donor_iface_lddt":round(dm,3),
                 "best_leader_rank":bl['rank'],"best_leader_dockq":bl['dockq_gt'],
                 "best_leader_iface_lddt":round(lm,3),
                 "dockq_loss":round(d['donor_dockq']-bl['dockq_gt'],3),
                 "lddt_loss":round(dm-lm,3),
                 "dockqdonor_rescued":dq[lab]['rescued'],
                 "leader20_rescued":ld.get(lab,{}).get('method_rescued')})
    r=rows[-1]
    print(f"{lab:11} | {r['dq_donor_dockq']:6.3f} {r['dq_donor_iface_lddt']:10.3f} "
          f"{'R' if r['dockqdonor_rescued'] else '':>6} | {r['best_leader_rank']:4d} "
          f"{r['best_leader_dockq']:6.3f} {r['best_leader_iface_lddt']:10.3f} "
          f"{'R' if r['leader20_rescued'] else '':>4} | {r['dockq_loss']:+6.3f} {r['lddt_loss']:+5.3f}")
json.dump(rows,open(f"{EXP}/artifacts/donor_vs_bestleader.json","w"),indent=1)
print(f"\n  median DockQ lost by using the best LEADER instead of the best pose: "
      f"{st.median(r['dockq_loss'] for r in rows):+.3f}")
print(f"  median interface lDDT lost:                                        "
      f"{st.median(r['lddt_loss'] for r in rows):+.3f}")
disc=[r for r in rows if r['dockqdonor_rescued'] and not r['leader20_rescued']]
if disc:
    print(f"\n  on the {len(disc)} interfaces dockqdonor rescues and leader20 does not:")
    print(f"    median DockQ loss {st.median(r['dockq_loss'] for r in disc):+.3f}   "
          f"median lDDT loss {st.median(r['lddt_loss'] for r in disc):+.3f}")
