#!/usr/bin/env python
"""Sharper donor metrics: framework-aligned CDR-H3 backbone RMSD, and CA accuracy at CONTACT residues.

lDDT over interface residues separated the dockqdonor and topiptm donors by only ~0.03 (median
paratope 0.870 vs 0.843, epitope 0.888 vs 0.853) against a DockQ difference of 0.486 vs 0.036. lDDT is
a superposition-free, distance-distribution measure and may be insensitive to a rigid displacement or
a rotated loop. Two sharper metrics:

  1. CDR-H3 backbone RMSD after FRAMEWORK alignment -- superpose the model heavy chain onto the native
     on VH framework CA atoms only (all VH residues except H3), then measure H3 CA RMSD. This is the
     standard antibody loop metric and is sensitive to loop placement, which lDDT partially forgives.
     H3 definition and framework/H3 split reuse scripts/h3_rmsd.py (C...WG.G motif, VH end = H3 end+15).

  2. CA RMSD restricted to ACTUAL CONTACT RESIDUES -- superpose the model chain onto the native chain
     on all CA, then measure RMSD over exactly the residues within 5 A of the partner in the native
     complex, per chain (paratope side and epitope side separately).

Both are computed for the SAME two donor poses compared in scripts/donor_fold_table.py.

Usage: python scripts/donor_h3_contact.py
Writes: artifacts/donor_h3_contact.json
"""
import json, os, sys, re, difflib, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gemmi
from decoy_set import PRED, EXP, native_chains
from fold_quality import TLC, nat_res, kind

dq=json.load(open(f"{EXP}/artifacts/dockq_arm_donors.json"))
ti=json.load(open(f"{EXP}/artifacts/topiptm26_donors.json"))

def seq_of(rs): return "".join(TLC[r.name] for r in rs)
def ca(r):
    a=r.find_atom("CA","*"); return a.pos if a else None
def find_h3(seq):                      # identical to scripts/h3_rmsd.py
    for m in re.finditer(r'WG.G', seq):
        w=m.start(); c=seq.rfind('C',0,w)
        if c==-1: continue
        if 1<=w-(c+1)<=35: return c+1,w
    return None
def model_chain(st,target):
    best,br=None,-1
    for ch in st[0]:
        s=seq_of([r for r in ch if r.name in TLC])
        if len(s)<4: continue
        r=difflib.SequenceMatcher(None,s,target,autojunk=False).ratio()
        if r>br: br,best=r,ch
    return best
def iface_idx(pdb,ch,partner):
    D=native_chains(pdb); XA,IA=D[ch]; XB,_=D[partner]
    order={v:i for i,v in enumerate(sorted(set(IA),key=lambda x:int(x)))}
    m=np.linalg.norm(XA[:,None,:]-XB[None,:,:],axis=-1)<5.0
    return {order[IA[i]] for i in np.where(m.any(1))[0]}

def metrics(pdb,ch_letter,partner,s,k):
    """(h3_rmsd or None, contact CA RMSD, n contacts, framework RMSD or None)."""
    nres=nat_res(pdb,ch_letter); nseq=seq_of(nres)
    cif=f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif): return None,None,0,None
    stm=gemmi.read_structure(cif); stm.setup_entities()
    ch=model_chain(stm,nseq)
    if ch is None: return None,None,0,None
    mres=[r for r in ch if r.name in TLC]; mseq=seq_of(mres)
    sm=difflib.SequenceMatcher(None,mseq,nseq,autojunk=False)
    pairs=[(a+i,b+i) for a,b,n in sm.get_matching_blocks() for i in range(n)]
    contacts=iface_idx(pdb,ch_letter,partner)

    # --- contact-residue CA RMSD after WHOLE-CHAIN superposition ---
    allm=[];alln=[]
    for mi,ni in pairs:
        cm,cn=ca(mres[mi]),ca(nres[ni])
        if cm and cn: allm.append(cm); alln.append(cn)
    crms=None
    if len(allm)>=3:
        T=gemmi.superpose_positions(alln,allm).transform
        sq=[];
        for mi,ni in pairs:
            if ni not in contacts: continue
            cm,cn=ca(mres[mi]),ca(nres[ni])
            if not (cm and cn): continue
            p=gemmi.Position(T.apply(cm)); sq.append(p.dist(cn)**2)
        if sq: crms=(sum(sq)/len(sq))**0.5

    # --- framework-aligned H3 RMSD (heavy chains only) ---
    h3r=fwr=None
    h3=find_h3(mseq)
    if h3:
        a,b=h3; vh_end=min(len(mseq),b+15)
        h3i=set(range(a,b)); fwi=set(range(0,vh_end))-h3i
        fm=[];fn=[];hm=[];hn=[]
        for mi,ni in pairs:
            cm,cn=ca(mres[mi]),ca(nres[ni])
            if not (cm and cn): continue
            if mi in fwi: fm.append(cm); fn.append(cn)
            elif mi in h3i: hm.append(cm); hn.append(cn)
        if len(fm)>=3 and hm:
            T=gemmi.superpose_positions(fn,fm)
            fwr=T.rmsd; Tt=T.transform
            sq=[gemmi.Position(Tt.apply(p)).dist(q)**2 for p,q in zip(hm,hn)]
            h3r=(sum(sq)/len(sq))**0.5
    return h3r,crms,len(contacts),fwr

rows=[]
print(f"{'interface':11} {'donor':>10} {'DockQ':>6} | {'H3 RMSD':>8} {'fw RMSD':>8} | "
      f"{'paratope CA RMSD':>16} {'epitope CA RMSD':>15}")
for lab in sorted(set(dq)&set(ti)):
    d=dq[lab]; t=ti[lab]; pdb=d['pdb']; c1,c2=d['interface']
    seqs={c:seq_of(nat_res(pdb,c)) for c in (c1,c2)}
    roles={c:kind(seqs[c]) for c in (c1,c2)}
    ag=[c for c in (c1,c2) if roles[c]=='ANTIGEN']
    if len(ag)!=1: continue
    ag=ag[0]; ab=[c for c in (c1,c2) if c!=ag][0]
    # H3 lives on the HEAVY chain. On 13 of 26 interfaces the antibody chain under test is the LIGHT
    # chain, so H3 is absent from it -- but the complex always contains a heavy chain. Measure H3 on
    # the Fab's heavy chain: the heavy chain with the most native contacts to the tested antibody
    # chain (H and L are paired and adjacent), falling back to the tested chain if it is itself heavy.
    hv = ab if roles[ab]=='HEAVY' else None
    if hv is None:
        D=native_chains(pdb); cands=[]
        for c,(X,I) in D.items():
            if c in (c1,c2): continue
            sq="".join(TLC[r.name] for r in nat_res(pdb,c)) if True else ""
            if len(sq)<90 or kind(sq)!='HEAVY': continue
            XA,_=D[ab]
            n=int((np.linalg.norm(XA[:,None,:]-X[None,:,:],axis=-1)<5.0).sum())
            cands.append((n,c))
        if cands: hv=max(cands)[1]
    rec={'interface':lab,'ab_role':roles[ab],'h3_chain':hv,
         'h3_chain_is_tested':bool(hv==ab)}
    for nm,src in (('dockqdonor',d),('topiptm',t)):
        _,cr_ab,n_ab,_=metrics(pdb,ab,ag,src['seed'],src['sample'])
        h3=fw=None
        if hv:
            h3,_,_,fw=metrics(pdb,hv,ag,src['seed'],src['sample'])
        _,cr_ag,n_ag,_=metrics(pdb,ag,ab,src['seed'],src['sample'])
        rec[nm]={'dockq':src.get('donor_dockq'),'h3_rmsd':round(h3,3) if h3 else None,
                 'fw_rmsd':round(fw,3) if fw else None,
                 'paratope_ca_rmsd':round(cr_ab,3) if cr_ab else None,
                 'epitope_ca_rmsd':round(cr_ag,3) if cr_ag else None,
                 'n_paratope':n_ab,'n_epitope':n_ag}
        v=rec[nm]
        f3=f"{v['h3_rmsd']:8.2f}" if v['h3_rmsd'] else f"{'-':>8}"
        ff=f"{v['fw_rmsd']:8.2f}" if v['fw_rmsd'] else f"{'-':>8}"
        pa=f"{v['paratope_ca_rmsd']:16.2f}" if v['paratope_ca_rmsd'] else f"{'-':>16}"
        ep=f"{v['epitope_ca_rmsd']:15.2f}" if v['epitope_ca_rmsd'] else f"{'-':>15}"
        print(f"{lab:11} {nm:>10} {(v['dockq'] or 0):6.3f} | {f3} {ff} | {pa} {ep}", flush=True)
    rows.append(rec)
json.dump(rows,open(f"{EXP}/artifacts/donor_h3_contact.json","w"),indent=1)

def med(nm,f):
    v=[r[nm][f] for r in rows if r[nm].get(f) is not None]
    return st.median(v) if v else float('nan')
print(f"\nMEDIANS  (n={len(rows)} interfaces; H3 only where a heavy chain is in the interface)")
for f,lbl in (('h3_rmsd','CDR-H3 RMSD (A)'),('fw_rmsd','framework RMSD (A)'),
              ('paratope_ca_rmsd','paratope CA RMSD (A)'),('epitope_ca_rmsd','epitope CA RMSD (A)')):
    a,b=med('dockqdonor',f),med('topiptm',f)
    n=sum(1 for r in rows if r['dockqdonor'].get(f) is not None and r['topiptm'].get(f) is not None)
    print(f"  {lbl:24} dockqdonor {a:6.2f}   topiptm {b:6.2f}   diff {b-a:+6.2f}   (n={n})")
for f in ('h3_rmsd','paratope_ca_rmsd','epitope_ca_rmsd'):
    p=[r for r in rows if r['dockqdonor'].get(f) is not None and r['topiptm'].get(f) is not None]
    w=sum(1 for r in p if r['dockqdonor'][f]<r['topiptm'][f])
    print(f"  dockqdonor better on {f}: {w}/{len(p)}")
