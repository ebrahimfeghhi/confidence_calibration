#!/usr/bin/env python
"""Does the model ever sample a native-matching CDR-H3? (selection vs generation for the antibody)

For each antibody HEAVY chain (identified by kind() motif — immune to the agonly_antigen_idx bug),
score every pose's H3 loop lDDT vs native over the canonical pool (seeds 101-150, 250 poses):
  - oracle_h3lddt  = BEST-of-250 H3 lDDT  -> does a native-like H3 EXIST in the pool?
  - median_h3lddt  = typical pose's H3
  - wholeplddt_pick_h3lddt = H3 lDDT of the fold picked by whole-chain pLDDT (naive selection)
  - h3plddt_pick_h3lddt    = H3 lDDT of the fold picked by H3-REGION pLDDT (local selection)
Interpretation: oracle_h3lddt HIGH + naive-pick LOW => SELECTION problem (native H3 in pool, pLDDT
misses it) -> an H3-local measure is worth building. oracle_h3lddt LOW => GENERATION problem
(model never draws native H3) -> no selector helps, must sample H3 harder.
CDR-H3 = residues between the FR3 Cys and the FR4 WGxG motif. Run with Protenix venv python (gemmi).
"""
import gemmi, subprocess, os, json, glob, statistics, difflib, re
from concurrent.futures import ThreadPoolExecutor

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
GT   = "/home/ebrahim/FoldBench/examples/ground_truths"
NATD = "/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl/native_chains"
OST  = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
OUT  = f"{EXP}/artifacts/h3_inpool"; os.makedirs(f"{OUT}/chains", exist_ok=True); os.makedirs(f"{OUT}/ost", exist_ok=True)
ENV  = dict(os.environ); ENV["LD_LIBRARY_PATH"] = "/home/ebrahim/miniconda3/envs/foldbench/lib"
chainmap = json.load(open(f"{EXP}/artifacts/oracle_chainmap.json"))   # qidx -> native letter (correct)

TLC = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)
def kind(s):
    hasW=re.search(r'WG.G[TS]',s);hasF=re.search(r'FG.G[TS]',s);w=s[:24]
    ntL=re.search(r'([DE]I[QV][ML]TQ|[DE]VVMTQ|EIVLTQ|QSVLTQ|DIQLTQ|SYELTQ|DIVMTQ)',w)
    ntH=re.search(r'([QE][VL][QK]L[VLEQ][EQ]S|QVKLEES|VESGG[GA]|VQSGAE|VETGGG|QESGGG)',w)
    if not (95<=len(s)<=275): return 'ANTIGEN'
    light=bool(ntL) or (bool(hasF) and not hasW); heavy=bool(ntH) or bool(hasW)
    if not(light or heavy): return 'ANTIGEN'
    if light and not ntH: return 'LIGHT'
    return 'HEAVY'
def find_h3(seq):
    for m in re.finditer(r'WG.G', seq):
        w=m.start(); c=seq.rfind('C',0,w)
        if c==-1: continue
        if 1<=w-(c+1)<=35: return c+1, w   # H3 = seq[c+1:w]
    return None
def best_match_chain(st, target):
    best, br = None, -1
    for ch in st[0]:
        s=seqof(ch)
        if len(s)<4: continue
        r=difflib.SequenceMatcher(None, s, target, autojunk=False).ratio()
        if r>br: br, best = r, ch
    return best
def run_ost(m, n, o):
    # delegates to eval/ost_dockq.py -- the single OST entry point. This used to be a
    # local copy; the copies drifted and five of them still passed --rigid-scores, which
    # crashes on small interfaces and loses the whole score.
    return _ost_run(m, n, o)

def poses(pdb):
    out=[]
    for s in range(101,151): out += sorted(glob.glob(f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_*.cif"))
    return out

def native_ref(pdb, letter):
    p=f"{NATD}/{pdb}_{letter}.pdb"
    if os.path.exists(p): return p
    st=gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif"); st.setup_entities()
    ch=next((c for c in st[0] if c.name==letter), None)
    p=f"{OUT}/chains/{pdb}_{letter}_native.pdb"
    ns=gemmi.Structure(); ns.add_model(gemmi.Model("1")); ns[0].add_chain(ch); ns.setup_entities(); ns.write_pdb(p)
    return p

def score_pose(args):
    idx, cif, hseq, natref, tag = args
    try: st=gemmi.read_structure(cif); st.setup_entities()
    except Exception: return None
    ch=best_match_chain(st, hseq)
    if ch is None: return None
    mseq=seqof(ch); h3=find_h3(mseq)
    if h3 is None: return None
    s,e=h3
    resnums=[r.seqid.num for r in ch if r.name in TLC]
    h3_resnums=set(resnums[s:e])
    # pLDDT (b_iso/100)
    def plddt(residset=None):
        vals=[at.b_iso for r in ch for at in r if (residset is None or r.seqid.num in residset)]
        return statistics.mean(vals)/100.0 if vals else None
    chain_pl=plddt(); h3_pl=plddt(h3_resnums)
    mp=f"{OUT}/chains/{tag}_{idx}.pdb"
    ns=gemmi.Structure(); ns.add_model(gemmi.Model("1")); ns[0].add_chain(ch); ns.setup_entities(); ns.write_pdb(mp)
    o=f"{OUT}/ost/{tag}_{idx}.json"
    run_ost(mp, natref, o)
    try: os.remove(mp)
    except Exception: pass
    try: loc=json.load(open(o)).get("local_lddt")
    except Exception: loc=None
    if not loc: return None
    h3vals=[]
    for k,v in loc.items():
        if v is None: continue
        try: rn=int(k.split('.')[1])
        except Exception: continue
        if rn in h3_resnums: h3vals.append(v)
    if not h3vals: return None
    return {"h3_lddt": round(statistics.mean(h3vals),3), "h3_plddt": round(h3_pl,3),
            "chain_plddt": round(chain_pl,3), "cif": os.path.relpath(cif, PRED)}

results=[]
for pdb in sorted(set(json.load(open(f"{EXP}/artifacts/meta.json"))[k]["pdb"] for k in json.load(open(f"{EXP}/artifacts/meta.json")))):
    base=json.load(open(f"{JDIR}/{pdb}-final-updated.json")); o=base[0] if isinstance(base,list) else base
    qseqs=[e["proteinChain"]["sequence"] for e in o["sequences"] if "proteinChain" in e]
    for hq,hseq in enumerate(qseqs):
        if kind(hseq)!='HEAVY': continue
        h3=find_h3(hseq)
        if h3 is None: continue
        natlet=chainmap[pdb].get(str(hq))
        if natlet is None: continue
        natref=native_ref(pdb, natlet); tag=f"{pdb}_H{hq}"
        args=[(i,c,hseq,natref,tag) for i,c in enumerate(poses(pdb))]
        scored=[x for x in ThreadPoolExecutor(12).map(score_pose, args) if x]
        if not scored:
            print(f"{tag}: no scored poses", flush=True); continue
        h3ls=[x["h3_lddt"] for x in scored]
        oracle=max(scored, key=lambda x:x["h3_lddt"])
        wpl=max(scored, key=lambda x:x["chain_plddt"])
        h3pl=max(scored, key=lambda x:x["h3_plddt"])
        row={"pdb":pdb,"heavy_qidx":hq,"native_chain":natlet,"h3_seq":hseq[h3[0]:h3[1]],"h3_len":h3[1]-h3[0],
             "n_poses":len(scored),
             "oracle_h3lddt":oracle["h3_lddt"], "median_h3lddt":round(statistics.median(h3ls),3),
             "min_h3lddt":min(h3ls),
             # oracle (best-H3-lDDT) pose's OWN confidence — is the best pose even confident?
             "oracle_chain_plddt":oracle["chain_plddt"], "oracle_h3_plddt":oracle["h3_plddt"],
             # pick by GLOBAL whole-chain pLDDT: the confidence used + the H3 lDDT you get
             "globalplddt_pick_chainplddt":wpl["chain_plddt"], "globalplddt_pick_h3lddt":wpl["h3_lddt"],
             # pick by H3-REGION pLDDT: the confidence used + the H3 lDDT you get
             "h3plddt_pick_h3plddt":h3pl["h3_plddt"], "h3plddt_pick_h3lddt":h3pl["h3_lddt"],
             "oracle_pose":oracle["cif"]}
        results.append(row)
        print(f"{tag:9s} H3len={row['h3_len']:2d} n={len(scored):3d} | best-H3lDDT={row['oracle_h3lddt']:.3f}"
              f"(its pLDDT: chain={oracle['chain_plddt']:.3f} H3={oracle['h3_plddt']:.3f}) | "
              f"GLOBALpLDDTpick pLDDT={wpl['chain_plddt']:.3f}->H3lDDT={wpl['h3_lddt']:.3f} | "
              f"H3pLDDTpick pLDDT={h3pl['h3_plddt']:.3f}->H3lDDT={h3pl['h3_lddt']:.3f}", flush=True)

json.dump(results, open(f"{OUT}/h3_inpool_results.json","w"), indent=1)
print("\nWROTE", f"{OUT}/h3_inpool_results.json")
