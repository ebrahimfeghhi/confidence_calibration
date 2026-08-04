#!/usr/bin/env python
"""PHASE 2 of the 7-run template sweep: build every template JSON, for both interface sets.

The sweep asks: if you template BOTH chains with a model-generated fold, does the criterion used to
pick that fold decide whether the confident decoy is rescued? Four criteria, all derived from the one
per-pose table written by fold_selection_table.py:

  globallddt  - best whole-chain lDDT vs native   (ORACLE; needs the answer, measures the ceiling)
  ifacelddt   - best paratope/epitope lDDT        (ORACLE, binding-site-local)
  globalplddt - best whole-chain pLDDT            (the ONLY genuinely DEPLOYABLE arm)
  ifaceplddt  - best paratope/epitope pLDDT       (ORACLE: pLDDT is model-internal, but the
                paratope/epitope are defined from the NATIVE complex, which is unknown at
                deployment -- so this measures a ceiling, not a shippable method)

The other three arms of the sweep use NATIVE folds and already have template JSONs built:
  native50 (both chains native), abonly (antibody only), agonly (antigen only).
For those we only re-point the a3m paths at durable storage and re-run the pdbs whose predictions sit
at the wrong seed range.

Template mechanics (unchanged from build_pddag.py / README 6.3): the chosen pose is registered as a
fake-old mmCIF (release_date 2000-01-01) under a fresh tid, and referenced from a 2-line a3m; two
residues are mutated in the a3m hit so Protenix treats it as a template rather than an MSA row.
multichain_mask means a template carries ZERO cross-chain information.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_sweep_templates.py
Writes: artifacts/sweep_tmpl/<run>/<pdb>.json, artifacts/sweep_a3m/, artifacts/sweep_jobs.json
"""
import json, os, sys, glob, difflib, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED

JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
SCRATCH = ("/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad")
TMPL = f"{EXP}/artifacts/sweep_tmpl"; A3M = f"{EXP}/artifacts/sweep_a3m"
NATIVE_A3M = f"{EXP}/artifacts/native_a3m_all"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
SEEDS = list(range(101, 151))
CRITERIA = {"globallddt": 0, "ifacelddt": 1, "globalplddt": 2, "ifaceplddt": 3}   # index into pose row
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
tbl = json.load(open(f"{EXP}/artifacts/fold_selection_table.json"))
cache = json.load(open(CACHE_F))
DEC = json.load(open(f"{EXP}/artifacts/decoy_set_101_150.json"))
DECOY_PDBS = sorted({DEC["per_interface"][l]["pdb"] for l in DEC["decoys"]})
GOOD_PDBS = sorted({e["pdb"] for e in json.load(open(f"{EXP}/artifacts/safety15.json"))})
ALL_PDBS = sorted(set(DECOY_PDBS) | set(GOOD_PDBS))     # 26: 11 decoy pdbs + 15 good pdbs


def setof(pdb):
    """Explicit set membership; raises rather than silently labelling an unknown pdb."""
    if pdb in DECOY_PDBS: return "decoy"
    if pdb in GOOD_PDBS: return "good"
    raise KeyError(f"{pdb} is in neither the decoy nor the good set")


def used_tids():
    return {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}


TAKEN = used_tids()
_n = [0]


def new_tid():
    while True:
        _n[0] += 1
        t = f"w{_n[0]:03d}"
        if t not in TAKEN:
            TAKEN.add(t); return t


def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)
def mut(a): return "A" if a != "A" else "G"


def best_chain(st, target):
    best, br = None, -1
    for ch in st[0]:
        s = seqof(ch)
        if len(s) < 4: continue
        r = difflib.SequenceMatcher(None, s, target, autojunk=False).ratio()
        if r > br: br, best = r, ch
    return best


def register(cif_path, cif2tid):
    """Register a predicted pose as a fake-old mmCIF template; returns its tid."""
    if cif_path in cif2tid: return cif2tid[cif_path]
    tid = new_tid()
    cache.setdefault(tid, {"release_date": "2000-01-01"})
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif_path).read(), tid))
    cif2tid[cif_path] = tid
    return tid


jobs = collections.defaultdict(list)

# ---------- the four predicted-fold criteria: BOTH chains templated ----------
for run, col in CRITERIA.items():
    os.makedirs(f"{TMPL}/{run}", exist_ok=True)
    for pdb in ALL_PDBS:
        if pdb not in tbl:
            print(f"SKIP {run} {pdb}: not in fold_selection_table"); continue
        base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
        obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
        qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
        cif2tid, pi, info, ok = {}, 0, [], True
        for e in obj["sequences"]:
            if "proteinChain" not in e: continue
            rec = tbl[pdb].get(str(pi))
            if rec is None:                      # chain not part of a labelled interface -> no template
                e["proteinChain"].pop("templatesPath", None); info.append(f"q{pi}:free"); pi += 1; continue
            poses = [p for p in rec["poses"] if p[2 + col] is not None]
            if not poses:
                print(f"SKIP {run} {pdb} q{pi}: no pose has criterion {run}"); ok = False; break
            s, k, *vals = max(poses, key=lambda p: p[2 + col])
            cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
            st = gemmi.read_structure(cif); st.setup_entities()
            match = best_chain(st, qseqs[pi]).name
            tid = register(cif, cif2tid)
            q = qseqs[pi]; L = len(q); hs = list(q)
            for pos in (L // 3, 2 * L // 3): hs[pos] = mut(hs[pos])
            a3mp = f"{A3M}/{run}_{pdb}_q{pi}.a3m"
            open(a3mp, "w").write(
                f">query\n{q}\n>{tid}_{match}/1-{L} [subseq from] mol:protein length:{L}  {run}\n"
                f"{''.join(hs)}\n")
            e["proteinChain"]["templatesPath"] = a3mp
            info.append(f"q{pi}:seed{s}/s{k}={vals[col]:.3f}")
            pi += 1
        if not ok: continue
        obj["name"] = f"{pdb}_{run}"
        jp = f"{TMPL}/{run}/{pdb}.json"
        json.dump([obj], open(jp, "w"))
        jobs[run].append({"pdb": pdb, "json": jp, "name": obj["name"], "seeds": SEEDS,
                          "set": setof(pdb)})
        print(f"{run:12} {pdb}: {' '.join(info)}", flush=True)

# ---------- native-fold arms: reuse existing JSONs, re-point a3m to durable storage ----------
# Only the pdbs whose predictions sit at the WRONG seed range need re-running. 8ux6 is dropped: it is
# in neither final set (not a confident decoy at seeds 101-150, and not a good). 8fbw/8yxi are absent
# from agonly here because the label-corrected re-runs already exist at 101-150 in artifacts/relabel_out
# -- using the stale agonly_tmpl JSONs for them would REINTRODUCE the antigen-label bug.
NATIVE = {"native50": (NATIVE_A3M, "-nativetmpl.json"),
          "agonly":   (f"{SCRATCH}/selftmpl/agonly_tmpl", "-agonly.json")}
MISSING = {"native50": ["8fbw", "8q3j", "8yxi"], "agonly": ["8q3j"]}
for run, (src, suffix) in NATIVE.items():
    os.makedirs(f"{TMPL}/{run}", exist_ok=True)
    for pdb in MISSING[run]:
        cands = glob.glob(f"{src}/{pdb}{suffix}") or glob.glob(f"{src}/{pdb}*.json")
        if not cands:
            print(f"SKIP {run} {pdb}: no source template JSON found under {src}"); continue
        obj = json.load(open(cands[0])); obj = obj[0] if isinstance(obj, list) else obj
        for e in obj["sequences"]:
            if "proteinChain" not in e: continue
            tp = e["proteinChain"].get("templatesPath")
            if tp:                              # /tmp/.../tmpl/all/<pdb>_q<N>.a3m -> durable copy
                nb = f"{NATIVE_A3M}/{os.path.basename(tp)}"
                if not os.path.exists(nb):
                    print(f"  WARN {run} {pdb}: durable a3m missing for {os.path.basename(tp)}")
                e["proteinChain"]["templatesPath"] = nb
        obj["name"] = f"{pdb}_{run}"
        jp = f"{TMPL}/{run}/{pdb}.json"
        json.dump([obj], open(jp, "w"))
        jobs[run].append({"pdb": pdb, "json": jp, "name": obj["name"], "seeds": SEEDS,
                          "set": setof(pdb)})
        print(f"{run:12} {pdb}: reused {os.path.basename(cands[0])}, a3m -> durable", flush=True)

json.dump(cache, open(CACHE_F, "w"))
flat = [dict(run=r, **j) for r, js in jobs.items() for j in js]
json.dump(flat, open(f"{EXP}/artifacts/sweep_jobs.json", "w"), indent=1)
print(f"\nBUILD DONE: {len(flat)} jobs")
for r, js in sorted(jobs.items()):
    print(f"  {r:12} {len(js):3d} jobs "
          f"({sum(1 for j in js if j['set']=='decoy')} decoy / {sum(1 for j in js if j['set']=='good')} good)")
