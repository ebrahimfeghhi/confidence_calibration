#!/usr/bin/env python
"""Fold quality of the two donor folds -- best-DockQ (dockqdonor) vs top-ipTM (topiptm26).

For every confident-decoy interface, measures both donor poses against the native:
    global lDDT     whole chain
    paratope lDDT   lDDT over the antibody chain's residues within 5 A of the antigen
    epitope lDDT    lDDT over the antigen chain's residues within 5 A of the antibody

Chain roles come from fold_quality.kind() (HEAVY / LIGHT / ANTIGEN); interface residues are those of
the interface UNDER TEST only, not a union over a chain's interfaces.

Usage: python scripts/donor_fold_table.py
Writes: artifacts/donor_fold_table.json
"""
import json, os, sys, subprocess, difflib, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gemmi
from decoy_set import OST, ENV, PRED, EXP, native_chains
from fold_quality import TLC, nat_res, nat_pdb, kind

W=f"{EXP}/artifacts/donor_folds"; os.makedirs(W, exist_ok=True)
dq=json.load(open(f"{EXP}/artifacts/dockq_arm_donors.json"))
ti=json.load(open(f"{EXP}/artifacts/topiptm26_donors.json"))

def iface_idx(pdb,ch,partner):
    D=native_chains(pdb); XA,IA=D[ch]; XB,_=D[partner]
    order={v:i for i,v in enumerate(sorted(set(IA),key=lambda x:int(x)))}
    m=np.linalg.norm(XA[:,None,:]-XB[None,:,:],axis=-1)<5.0
    return {order[IA[i]] for i in np.where(m.any(1))[0]}

def lddt(pdb,ch,partner,s,k,tag):
    """(whole-chain lDDT, interface-local lDDT) for one chain of one pose."""
    seq="".join(TLC[r.name] for r in nat_res(pdb,ch)); ref=nat_pdb(pdb,ch)
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif): return None,None
    st_=gemmi.read_structure(cif); st_.setup_entities()
    best,br=None,-1
    for c in st_[0]:
        rs=[r for r in c if r.name in TLC]
        if not rs: continue
        r_=difflib.SequenceMatcher(None,"".join(TLC[x.name] for x in rs),seq,autojunk=False).ratio()
        if r_>br: br,best=r_,rs
    if best is None: return None,None
    mp=f"{W}/m_{tag}.pdb"
    ns=gemmi.Structure(); ns.add_model(gemmi.Model("1")); cc=gemmi.Chain("A")
    for r in best: cc.add_residue(r)
    ns[0].add_chain(cc); ns.setup_entities(); ns.write_pdb(mp)
    o=f"{W}/ld_{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset","-c","0-63",OST,"compare-structures","-m",mp,"-r",ref,"-o",o,
                        "--fault-tolerant","--min-pep-length","4","--lddt","--local-lddt"],
                       env=ENV,capture_output=True,timeout=600)
    try: d=json.load(open(o))
    except Exception: return None,None
    g=d.get("lddt"); ll=d.get("local_lddt") or {}; aln=(d.get("aln") or [None])[0]
    if not aln: return g,None
    p=aln.split("\n"); ri=mi=0; per={}
    for a,b in zip(p[1],p[3]):
        if a!="-" and b!="-":
            v=ll.get(f"A.{mi+1}.")
            if v is not None: per[ri]=v
        if a!="-": ri+=1
        if b!="-": mi+=1
    ii=iface_idx(pdb,ch,partner); vals=[per[i] for i in ii if i in per]
    return g,(st.mean(vals) if vals else None)

rows=[]
print(f"{'interface':11} {'donor':>10} {'DockQ':>6} | {'Ab global':>9} {'paratope':>8} | "
      f"{'Ag global':>9} {'epitope':>7}")
for lab in sorted(set(dq)&set(ti)):
    d=dq[lab]; t=ti[lab]; pdb=d['pdb']; c1,c2=d['interface']
    seqs={c:"".join(TLC[r.name] for r in nat_res(pdb,c)) for c in (c1,c2)}
    roles={c:kind(seqs[c]) for c in (c1,c2)}
    ag=[c for c in (c1,c2) if roles[c]=='ANTIGEN']
    if len(ag)!=1: continue
    ag=ag[0]; ab=[c for c in (c1,c2) if c!=ag][0]
    rec={'interface':lab,'pdb':pdb,'antibody_chain':ab,'antigen_chain':ag,
         'ab_role':roles[ab]}
    for nm,src in (('dockqdonor',d),('topiptm',t)):
        s,k=src['seed'],src['sample']
        abg,abi=lddt(pdb,ab,ag,s,k,f"{lab}_{nm}_ab")
        agg,agi=lddt(pdb,ag,ab,s,k,f"{lab}_{nm}_ag")
        rec[nm]={'seed':s,'sample':k,'dockq':src.get('donor_dockq'),
                 'ab_global':round(abg,3) if abg else None,'paratope':round(abi,3) if abi else None,
                 'ag_global':round(agg,3) if agg else None,'epitope':round(agi,3) if agi else None}
        v=rec[nm]
        print(f"{lab:11} {nm:>10} {(v['dockq'] or 0):6.3f} | "
              f"{(v['ab_global'] or 0):9.3f} {(v['paratope'] or 0):8.3f} | "
              f"{(v['ag_global'] or 0):9.3f} {(v['epitope'] or 0):7.3f}", flush=True)
    rows.append(rec)
json.dump(rows,open(f"{EXP}/artifacts/donor_fold_table.json","w"),indent=1)
def med(nm,f):
    v=[r[nm][f] for r in rows if r[nm][f] is not None]
    return st.median(v) if v else float('nan')
print(f"\n{'MEDIAN':11} {'dockqdonor':>10} {med('dockqdonor','dockq'):6.3f} | "
      f"{med('dockqdonor','ab_global'):9.3f} {med('dockqdonor','paratope'):8.3f} | "
      f"{med('dockqdonor','ag_global'):9.3f} {med('dockqdonor','epitope'):7.3f}")
print(f"{'':11} {'topiptm':>10} {med('topiptm','dockq'):6.3f} | "
      f"{med('topiptm','ab_global'):9.3f} {med('topiptm','paratope'):8.3f} | "
      f"{med('topiptm','ag_global'):9.3f} {med('topiptm','epitope'):7.3f}")
