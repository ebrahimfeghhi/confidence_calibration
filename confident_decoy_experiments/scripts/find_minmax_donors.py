#!/usr/bin/env python
"""Donor = the pose minimising MAX(paratope RMSD, epitope RMSD), ties broken on the sum.

WHY MIN-MAX AND NOT SUM (user, 2026-08-06): the native-templating requirement is conjunctive -- both
sides of the interface have to be right. A sum lets one side pay for the other: 8fbw_BC's
paratope-ranked donor sums to 7.41 A (paratope 0.89 + epitope 6.52) and would look acceptable, while
the max correctly reports 6.52 and rejects it. That donor produced 14 acceptable poses against
dockqdonor's 212.

WHY NO WEIGHTING: both quantities are CA RMSD in angstroms over interface residues, so they are already
on one scale; and under a max the weighting question largely dissolves. Contact counts either side of a
5 A cutoff are roughly reciprocal, so contact-weighting would land near equal anyway.

TIE-BREAK: min-max produces ties among poses that share the same limiting side; break on the sum, so
among poses with the same worst-side RMSD the one with the better other side wins. Sort key is
(max, sum).

Diagnosis this comes from (artifacts/donor_epitope_check.json): ranking on paratope alone traded the
epitope away, and the trade was expensive exactly on the winnable interfaces -- median epitope penalty
+1.38 A on the 5 interfaces dockqdonor rescues and paratopedonor fails, versus +0.04 A on the 14 where
neither rescues.

CPU only: gemmi superposition, no OST, no GPU.

Usage: python scripts/find_minmax_donors.py
Writes: artifacts/minmax_donor_picks.json
"""
import json, os, sys, difflib, statistics as st
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gemmi
from decoy_set import EXP, PRED, SEEDS, SAMPLES, native_chains
from fold_quality import TLC, nat_res, kind

sc=json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
EXCLUDE={"7z2m_AJ","8udz_AC"}
DEC={k:v for k,v in sc.items() if v["free_status"]=="confident_decoy" and k not in EXCLUDE}

def seq_of(rs): return "".join(TLC[r.name] for r in rs)
def ca(r):
    a=r.find_atom("CA","*"); return a.pos if a else None
def iface_idx(pdb,ch,partner):
    D=native_chains(pdb); XA,IA=D[ch]; XB,_=D[partner]
    order={v:i for i,v in enumerate(sorted(set(IA),key=lambda x:int(x)))}
    m=np.linalg.norm(XA[:,None,:]-XB[None,:,:],axis=-1)<5.0
    return {order[IA[i]] for i in np.where(m.any(1))[0]}

def side_rmsd(mres,nres,contacts):
    sm=difflib.SequenceMatcher(None,seq_of(mres),seq_of(nres),autojunk=False)
    pairs=[(a+i,b+i) for a,b,n in sm.get_matching_blocks() for i in range(n)]
    allm=[];alln=[]
    for mi,ni in pairs:
        cm,cn=ca(mres[mi]),ca(nres[ni])
        if cm and cn: allm.append(cm); alln.append(cn)
    if len(allm)<3: return None
    T=gemmi.superpose_positions(alln,allm).transform
    sq=[]
    for mi,ni in pairs:
        if ni not in contacts: continue
        cm,cn=ca(mres[mi]),ca(nres[ni])
        if cm and cn: sq.append(gemmi.Position(T.apply(cm)).dist(cn)**2)
    return (sum(sq)/len(sq))**0.5 if sq else None

def one(args):
    """(seed, sample, paratope RMSD, epitope RMSD) for one pose."""
    pdb,ab,ag,c_ab,c_ag,s,k=args
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif): return None
    try:
        stm=gemmi.read_structure(cif); stm.setup_entities()
    except Exception: return None
    chains=[[r for r in c if r.name in TLC] for c in stm[0]]
    chains=[c for c in chains if len(c)>=4]
    out={}
    for letter,contacts in ((ab,c_ab),(ag,c_ag)):
        nres=nat_res(pdb,letter); nseq=seq_of(nres)
        best,br=None,-1
        for rs in chains:
            r_=difflib.SequenceMatcher(None,seq_of(rs),nseq,autojunk=False).ratio()
            if r_>br: br,best=r_,rs
        if best is None: return None
        out[letter]=side_rmsd(best,nres,contacts)
    if out[ab] is None or out[ag] is None: return None
    return (s,k,out[ab],out[ag])

picks={}
for lab in sorted(DEC):
    v=DEC[lab]; pdb=v["pdb"]
    parts=v["chains"].split("-")
    c1,c2=(parts[0],"-".join(parts[1:])) if len(parts)>2 else (parts[0],parts[1])
    seqs={c:seq_of(nat_res(pdb,c)) for c in (c1,c2)}
    ag=[c for c in (c1,c2) if kind(seqs[c])=="ANTIGEN"]
    if len(ag)!=1:
        print(f"{lab}: no single antigen chain -- skipped"); continue
    ag=ag[0]; ab=[c for c in (c1,c2) if c!=ag][0]
    c_ab=iface_idx(pdb,ab,ag); c_ag=iface_idx(pdb,ag,ab)
    args=[(pdb,ab,ag,c_ab,c_ag,s,k) for s in SEEDS for k in SAMPLES]
    with ProcessPoolExecutor(12) as ex:
        res=[r for r in ex.map(one,args,chunksize=8) if r]
    if not res:
        print(f"{lab}: no poses measured -- skipped"); continue
    # MIN-MAX, tie-break on the sum
    s,k,pa,ep=min(res,key=lambda x:(max(x[2],x[3]), x[2]+x[3]))
    picks[lab]={"pdb":pdb,"interface":[c1,c2],"antibody_chain":ab,"antigen_chain":ag,
                "seed":s,"sample":k,"paratope_rmsd":round(pa,3),"epitope_rmsd":round(ep,3),
                "worst_side":round(max(pa,ep),3),"sum":round(pa+ep,3),
                "n_poses":len(res),"n_paratope_res":len(c_ab),"n_epitope_res":len(c_ag),
                "median_worst_side":round(st.median(max(x[2],x[3]) for x in res),3)}
    print(f"{lab:11} worst-side {max(pa,ep):5.2f} A (para {pa:5.2f} / epi {ep:5.2f}) "
          f"at seed{s}/s{k}   pool median worst-side {st.median(max(x[2],x[3]) for x in res):5.2f}",
          flush=True)
    json.dump(picks,open(f"{EXP}/artifacts/minmax_donor_picks.json","w"),indent=1)
print(f"\nWROTE artifacts/minmax_donor_picks.json ({len(picks)} interfaces)")
