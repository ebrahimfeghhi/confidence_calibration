#!/usr/bin/env python
"""Wait until every PDB in the full-MSA run has N>=16 seeds, then report:
  - DockQ per interface (selection = top interface pair_iptm pose; rescued iff DockQ>=0.23)
  - Fold quality per chain (lDDT vs native, best & mean over all poses)
Compare against the canonical free-run (MSA-subsampled) baseline: 0/15 rescued."""
import json, os, glob, subprocess, time, statistics, difflib
from concurrent.futures import ThreadPoolExecutor
import gemmi

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
OUT = f"{EXP}/artifacts/fullmsa_out"
GT  = "/home/ebrahim/FoldBench/examples/ground_truths"
OST = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV = dict(os.environ); ENV["LD_LIBRARY_PATH"] = "/home/ebrahim/miniconda3/envs/foldbench/lib"
CACHE = f"{EXP}/artifacts/fullmsa_n4"; os.makedirs(CACHE, exist_ok=True)
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
meta  = json.load(open(f"{EXP}/artifacts/meta.json"))
agidx = json.load(open(f"{EXP}/artifacts/agonly_antigen_idx.json"))
PDBS  = sorted(set(m["pdb"] for m in meta.values()))
N = 4
TLC = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def std(c): return [r for r in c if r.name in TLC]
def seqof(rl): return "".join(TLC[r.name] for r in rl)
bd = {}
exec(compile(open(f"{CODE_ROOT}/scripts/binding_dependence.py").read().split("rows = []")[0], "bd", "exec"), bd)
resolve_unit = bd["resolve_unit"]

def done_seeds(p):
    return sorted(int(os.path.basename(d).split('_')[1]) for d in glob.glob(f"{OUT}/{p}/seed_*")
                  if os.path.exists(f"{d}/predictions/{p}_summary_confidence_sample_4.json"))

t0 = time.time()
while time.time() - t0 < 86000:
    ready = {p: done_seeds(p) for p in PDBS}
    k = sum(1 for v in ready.values() if len(v) >= N)
    if k == len(PDBS): break
    print(f"[{time.strftime('%H:%M')}] {k}/12 pdbs at N>={N}: " + str({p: len(v) for p, v in ready.items()}), flush=True)
    time.sleep(300)
ready = {p: done_seeds(p)[:N] for p in PDBS}
print(f"\nALL AT N={N}\n", flush=True)

def run_ost(m, r, o, extra):
    if os.path.exists(o) and os.path.getsize(o) > 0: return
    subprocess.run(["taskset","-c","0-31",OST,"compare-structures","-m",m,"-r",r,"-o",o,
                    "--fault-tolerant","--min-pep-length","4"] + extra,
                   env=ENV, capture_output=True, timeout=600)

def dockq_of(o, cf, c1, c2):
    try: d = json.load(open(o))
    except Exception: return None, None
    dq = mc = None
    for i, t in enumerate(d.get("dockq_interfaces") or []):
        if {t[0], t[1]} == {c1, c2}:
            q = d.get("dockq") or []; dq = q[i] if i < len(q) else None; mc = (t[2], t[3]); break
    pi = None
    try:
        c = json.load(open(cf)); M = c.get("chain_pair_iptm")
        if mc and M:
            a = ord(mc[0]) - ord('A'); b = ord(mc[1]) - ord('A')
            if 0 <= a < len(M) and 0 <= b < len(M): pi = M[a][b]
    except Exception: pass
    return dq, pi

dq_rows = []
for ifc, m in meta.items():
    p = m["pdb"]; nat = f"{GT}/{p}-assembly1.cif"
    def one(a, _p=p, _ifc=ifc, _m=m, _nat=nat):
        s, k = a
        cif = f"{OUT}/{_p}/seed_{s}/predictions/{_p}_sample_{k}.cif"
        cf  = f"{OUT}/{_p}/seed_{s}/predictions/{_p}_summary_confidence_sample_{k}.json"
        if not os.path.exists(cif): return None
        o = f"{CACHE}/dq_{_ifc}_{s}_{k}.json"; run_ost(cif, _nat, o, ["--dockq"])
        return dockq_of(o, cf, _m["c1"], _m["c2"])
    poses = []
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, [(s, k) for s in ready[p] for k in range(5)]):
            if r and r[0] is not None:
                poses.append((r[1] if r[1] is not None else -1, r[0]))
    if not poses: continue
    sel = max(poses, key=lambda x: x[0])
    dq_rows.append({"interface": ifc, "n_poses": len(poses), "sel_pair_iptm": round(sel[0], 3),
                    "sel_dockq": round(sel[1], 3), "best_dockq": round(max(d for _, d in poses), 3),
                    "rescued": sel[1] >= 0.23})
    print(f"DOCKQ {ifc:10} n={len(poses):3d} sel_ipTM={sel[0]:.3f} sel_DockQ={sel[1]:.3f} "
          f"best={max(d for _, d in poses):.3f} {'RESCUED' if sel[1] >= 0.23 else '-'}", flush=True)

fold = []
for p in PDBS:
    aidx = [int(x) for x in agidx[p]]
    d = json.load(open(f"{JDIR}/{p}-final-updated.json")); o_ = d[0] if isinstance(d, list) else d
    qseqs = [e["proteinChain"]["sequence"] for e in o_["sequences"] if "proteinChain" in e]
    letters, _ = resolve_unit(p, qseqs, aidx)
    for qi, qseq in enumerate(qseqs):
        L = letters.get(qi)
        if L is None: continue
        stn = gemmi.read_structure(f"{GT}/{p}-assembly1.cif"); stn.setup_entities()
        ch = next((c for c in stn[0] if c.name == L), None)
        if ch is None: continue
        nref = f"{CACHE}/nat_{p}_{L}.pdb"
        if not os.path.exists(nref):
            ns = gemmi.Structure(); ns.add_model(gemmi.Model("1")); ns[0].add_chain(ch)
            ns.setup_entities(); ns.write_pdb(nref)
        def fold_one(a, _p=p, _qi=qi, _qseq=qseq, _nref=nref):
            s, k = a
            cif = f"{OUT}/{_p}/seed_{s}/predictions/{_p}_sample_{k}.cif"
            if not os.path.exists(cif): return None
            try:
                stm = gemmi.read_structure(cif); stm.setup_entities()
            except Exception: return None
            best, br = None, -1
            for c in stm[0]:
                r = difflib.SequenceMatcher(None, seqof(std(c)), _qseq, autojunk=False).ratio()
                if r > br: br, best = r, c
            if best is None: return None
            mp = f"{CACHE}/m_{_p}_{_qi}_{s}_{k}.pdb"
            ms = gemmi.Structure(); ms.add_model(gemmi.Model("1")); ms[0].add_chain(best)
            ms.setup_entities(); ms.write_pdb(mp)
            oo = f"{CACHE}/ld_{_p}_{_qi}_{s}_{k}.json"; run_ost(mp, _nref, oo, ["--lddt"])
            try: v = json.load(open(oo)).get("lddt")
            except Exception: v = None
            try: os.remove(mp)
            except Exception: pass
            return v
        vals = []
        with ThreadPoolExecutor(6) as ex:
            for v in ex.map(fold_one, [(s, k) for s in ready[p] for k in range(5)]):
                if v is not None: vals.append(v)
        if vals:
            role = "ANTIGEN" if qi in aidx else "ANTIBODY"
            fold.append({"pdb": p, "qidx": qi, "role": role, "chain": L, "n": len(vals),
                         "best_lddt": round(max(vals), 3), "mean_lddt": round(statistics.mean(vals), 3)})
            print(f"FOLD {p} q{qi} {role:8} n={len(vals):3d} best={max(vals):.3f} "
                  f"mean={statistics.mean(vals):.3f}", flush=True)

json.dump({"N": N, "dockq": dq_rows, "fold": fold},
          open(f"{EXP}/artifacts/fullmsa_n4_report.json", "w"), indent=1)
resc = [r["interface"] for r in dq_rows if r["rescued"]]
print(f"\nRESCUED {len(resc)}/{len(dq_rows)} -> {resc}")
print("WROTE fullmsa_n4_report.json")
