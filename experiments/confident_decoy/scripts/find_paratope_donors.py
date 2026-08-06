#!/usr/bin/env python
"""Find, per interface, the free-run pose with the LOWEST paratope CA RMSD to native.

dockqdonor ranks candidate donors by DockQ (how well the pose docked). This ranks them by how well the
pose's PARATOPE is folded, measured as CA RMSD over the antibody chain's native contact residues after
whole-chain superposition -- the metric that separated the dockqdonor and topiptm donors (1.77 vs
2.58 A median) where lDDT did not (0.870 vs 0.843).

Both are ORACLES (they need the native), so this is a mechanism probe, not a deployable method.

CPU only: gemmi superposition, no OST, no GPU.

Usage: python scripts/find_paratope_donors.py
Writes: artifacts/paratope_donor_picks.json
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

def one(args):
    """paratope CA RMSD of one pose, or None."""
    pdb,ab,contacts,nseq,s,k=args
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif): return None
    try:
        st_=gemmi.read_structure(cif); st_.setup_entities()
    except Exception: return None
    best,br=None,-1
    for c in st_[0]:
        rs=[r for r in c if r.name in TLC]
        if len(rs)<4: continue
        r_=difflib.SequenceMatcher(None,seq_of(rs),nseq,autojunk=False).ratio()
        if r_>br: br,best=r_,rs
    if best is None: return None
    nres=nat_res(pdb,ab)
    sm=difflib.SequenceMatcher(None,seq_of(best),nseq,autojunk=False)
    pairs=[(a+i,b+i) for a,b,n in sm.get_matching_blocks() for i in range(n)]
    allm=[];alln=[]
    for mi,ni in pairs:
        cm,cn=ca(best[mi]),ca(nres[ni])
        if cm and cn: allm.append(cm); alln.append(cn)
    if len(allm)<3: return None
    T=gemmi.superpose_positions(alln,allm).transform
    sq=[]
    for mi,ni in pairs:
        if ni not in contacts: continue
        cm,cn=ca(best[mi]),ca(nres[ni])
        if cm and cn: sq.append(gemmi.Position(T.apply(cm)).dist(cn)**2)
    if not sq: return None
    return (s,k,(sum(sq)/len(sq))**0.5)

picks={}
for lab in sorted(DEC):
    v=DEC[lab]; pdb=v["pdb"]
    parts=v["chains"].split("-")
    c1,c2=(parts[0],"-".join(parts[1:])) if len(parts)>2 else (parts[0],parts[1])
    seqs={c:seq_of(nat_res(pdb,c)) for c in (c1,c2)}
    ag=[c for c in (c1,c2) if kind(seqs[c])=="ANTIGEN"]
    if len(ag)!=1:
        print(f"{lab}: cannot identify a single antigen chain -- skipped"); continue
    ag=ag[0]; ab=[c for c in (c1,c2) if c!=ag][0]
    contacts=iface_idx(pdb,ab,ag)
    args=[(pdb,ab,contacts,seqs[ab],s,k) for s in SEEDS for k in SAMPLES]
    with ProcessPoolExecutor(12) as ex:
        res=[r for r in ex.map(one,args,chunksize=8) if r]
    if not res:
        print(f"{lab}: no poses measured -- skipped"); continue
    s,k,r=min(res,key=lambda x:x[2])
    picks[lab]={"pdb":pdb,"interface":[c1,c2],"antibody_chain":ab,"antigen_chain":ag,
                "seed":s,"sample":k,"paratope_rmsd":round(r,3),
                "n_poses":len(res),"n_paratope_res":len(contacts),
                "median_paratope_rmsd":round(st.median(x[2] for x in res),3)}
    print(f"{lab:11} best paratope RMSD {r:5.2f} A at seed{s}/s{k}  "
          f"(median over {len(res)} poses {st.median(x[2] for x in res):5.2f})", flush=True)
    json.dump(picks,open(f"{EXP}/artifacts/paratope_donor_picks.json","w"),indent=1)
print(f"\nWROTE artifacts/paratope_donor_picks.json ({len(picks)} interfaces)")
