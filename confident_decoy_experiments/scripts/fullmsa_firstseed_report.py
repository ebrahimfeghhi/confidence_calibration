#!/usr/bin/env python
"""Wait until every PDB in the full-MSA run has >=1 completed seed, then report
DockQ (per interface) and fold quality (per-chain lDDT vs native) for that seed's 5 samples.
Compares against the canonical free-run (subsampled-MSA) baseline."""
import json, os, glob, subprocess, time, statistics, difflib, sys
from concurrent.futures import ThreadPoolExecutor
import gemmi

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
OUT=f"{EXP}/artifacts/fullmsa_out"; GT="/home/ebrahim/FoldBench/examples/ground_truths"
FREE="/home/ebrahim/Protenix/foldbench_abag/predictions_template"
OST="/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV=dict(os.environ); ENV["LD_LIBRARY_PATH"]="/home/ebrahim/miniconda3/envs/foldbench/lib"
CACHE=f"{EXP}/artifacts/fullmsa_firstseed"; os.makedirs(CACHE,exist_ok=True)
meta=json.load(open(f"{EXP}/artifacts/meta.json"))
agidx=json.load(open(f"{EXP}/artifacts/agonly_antigen_idx.json"))
JDIR="/home/ebrahim/Protenix/foldbench_abag/jsons"
PDBS=sorted(set(m["pdb"] for m in meta.values()))
TLC={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def std(c): return [r for r in c if r.name in TLC]
def seqof(rl): return "".join(TLC[r.name] for r in rl)
src=open(f"{CODE_ROOT}/scripts/binding_dependence.py").read().split("rows = []")[0]
bd={}; exec(compile(src,"bd","exec"), bd); resolve_unit=bd["resolve_unit"]

def done_seeds(p):
    return sorted(int(os.path.basename(d).split('_')[1]) for d in glob.glob(f"{OUT}/{p}/seed_*")
                  if os.path.exists(f"{d}/predictions/{p}_summary_confidence_sample_4.json"))
# ---- wait ----
t0=time.time()
while time.time()-t0 < 86000:
    ready={p:done_seeds(p) for p in PDBS}
    if all(v for v in ready.values()): break
    n=sum(1 for v in ready.values() if v)
    print(f"[{time.strftime('%H:%M')}] {n}/12 pdbs have a finished seed: "
          f"{ {k:(v[0] if v else 0) for k,v in ready.items()} }", flush=True)
    time.sleep(300)
ready={p:done_seeds(p) for p in PDBS}
print(f"\nALL READY. first finished seed per pdb: { {k:v[0] for k,v in ready.items()} }\n", flush=True)

def run_ost(m,r,o,extra):
    # delegates to eval/ost_dockq.py -- the single OST entry point. This used to be a
    # local copy; the copies drifted and five of them still passed --rigid-scores, which
    # crashes on small interfaces and loses the whole score.
    return _ost_run(m, r, o, extra=extra)

def dockq_of(o,cf,c1,c2):
    try: d=json.load(open(o))
    except Exception: return None,None
    dq=mc=None
    for k,t in enumerate(d.get("dockq_interfaces") or []):
        if {t[0],t[1]}=={c1,c2}:
            q=d.get("dockq") or []; dq=q[k] if k<len(q) else None; mc=(t[2],t[3]); break
    pi=None
    try:
        c=json.load(open(cf)); M=c.get("chain_pair_iptm")
        if mc and M:
            i=ord(mc[0])-ord('A'); j=ord(mc[1])-ord('A')
            if 0<=i<len(M) and 0<=j<len(M): pi=M[i][j]
    except Exception: pass
    return dq,pi

# ---- DockQ per interface (full-MSA seed) ----
rows=[]
for ifc,m in meta.items():
    p=m["pdb"]; s=ready[p][0]; nat=f"{GT}/{p}-assembly1.cif"
    poses=[]
    for k in range(5):
        cif=f"{OUT}/{p}/seed_{s}/predictions/{p}_sample_{k}.cif"
        cf=f"{OUT}/{p}/seed_{s}/predictions/{p}_summary_confidence_sample_{k}.json"
        if not os.path.exists(cif): continue
        o=f"{CACHE}/dq_{ifc}_{s}_{k}.json"; run_ost(cif,nat,o,["--dockq"])
        dq,pi=dockq_of(o,cf,m["c1"],m["c2"])
        if dq is not None: poses.append((pi if pi is not None else -1, dq))
    if not poses: continue
    sel=max(poses,key=lambda x:x[0])
    rows.append({"interface":ifc,"seed":s,"n":len(poses),"sel_pair_iptm":round(sel[0],3),
                 "sel_dockq":round(sel[1],3),"best_dockq":round(max(d for _,d in poses),3),
                 "rescued":sel[1]>=0.23})
    print(f"DOCKQ {ifc:10} seed{s} sel_ipTM={sel[0]:.3f} sel_DockQ={sel[1]:.3f} best={max(d for _,d in poses):.3f} "
          f"{'RESCUED' if sel[1]>=0.23 else '-'}", flush=True)

# ---- fold quality: per-chain lDDT ----
fold=[]
for p in PDBS:
    s=ready[p][0]
    aidx=[int(x) for x in agidx[p]]
    d=json.load(open(f"{JDIR}/{p}-final-updated.json")); o_=d[0] if isinstance(d,list) else d
    qseqs=[e["proteinChain"]["sequence"] for e in o_["sequences"] if "proteinChain" in e]
    letters,_=resolve_unit(p,qseqs,aidx)
    for qi,qseq in enumerate(qseqs):
        L=letters.get(qi)
        if L is None: continue
        stn=gemmi.read_structure(f"{GT}/{p}-assembly1.cif"); stn.setup_entities()
        ch=next((c for c in stn[0] if c.name==L),None)
        if ch is None: continue
        nref=f"{CACHE}/nat_{p}_{L}.pdb"
        if not os.path.exists(nref):
            ns=gemmi.Structure(); ns.add_model(gemmi.Model("1")); ns[0].add_chain(ch); ns.setup_entities(); ns.write_pdb(nref)
        vals=[]
        for k in range(5):
            cif=f"{OUT}/{p}/seed_{s}/predictions/{p}_sample_{k}.cif"
            if not os.path.exists(cif): continue
            stm=gemmi.read_structure(cif); stm.setup_entities()
            best,br=None,-1
            for c in stm[0]:
                r=difflib.SequenceMatcher(None,seqof(std(c)),qseq,autojunk=False).ratio()
                if r>br: br,best=r,c
            if best is None: continue
            mp=f"{CACHE}/m_{p}_{qi}_{k}.pdb"
            ms=gemmi.Structure(); ms.add_model(gemmi.Model("1")); ms[0].add_chain(best); ms.setup_entities(); ms.write_pdb(mp)
            oo=f"{CACHE}/ld_{p}_{qi}_{s}_{k}.json"; run_ost(mp,nref,oo,["--lddt"])
            try: v=json.load(open(oo)).get("lddt")
            except Exception: v=None
            if v is not None: vals.append(v)
            try: os.remove(mp)
            except Exception: pass
        if vals:
            role="ANTIGEN" if qi in aidx else "ANTIBODY"
            fold.append({"pdb":p,"qidx":qi,"role":role,"chain":L,"seed":s,
                         "best_lddt":round(max(vals),3),"mean_lddt":round(statistics.mean(vals),3)})
            print(f"FOLD {p} q{qi} {role:8} best_lDDT={max(vals):.3f} mean={statistics.mean(vals):.3f}", flush=True)
json.dump({"dockq":rows,"fold":fold},open(f"{EXP}/artifacts/fullmsa_firstseed_report.json","w"),indent=1)
print("\nWROTE fullmsa_firstseed_report.json")
