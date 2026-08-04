#!/usr/bin/env python
"""Build 'pddag' templates: best-pLDDT ANTIGEN fold as the ONLY template (antibody freed).
Deployable analog of agonly (native antigen only -> 8/15). Isolates whether a pLDDT-SELECTED
(non-native) antigen template alone rescues any confident decoy.

Convention (matches agonly exactly, only the antigen fold source differs):
  - antigen chain(s): templatesPath = custom a3m built from the best-pLDDT antigen fold
                      (pose registered as an mmCIF template, tid o940+; multichain_mask => no cross-chain info)
  - antibody chain(s): templatesPath = None  (no template — same state as the free run)
Antigen fold pick = highest mean-pLDDT antigen chain over seeds 101-150 (from localization_results.json).
Run with the Protenix venv python (has gemmi). See confident_decoy_README.md Methods section.
"""
import json, os, sys, difflib
sys.path.insert(0, "/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad")
from cifconv import convert
import gemmi

EXP  = "/home/ebrahim/confident_decoy_experiments"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPLDIR = f"{EXP}/artifacts/pddag_tmpl"; A3M = f"{EXP}/artifacts/pddag_a3m"
os.makedirs(TMPLDIR, exist_ok=True); os.makedirs(A3M, exist_ok=True)

agidx = json.load(open(f"{EXP}/artifacts/agonly_antigen_idx.json"))               # pdb -> [antigen qidx]
loc   = {(r['pdb'], r['antigen_qidx']): r
         for r in json.load(open(f"{EXP}/artifacts/antigen_localization/localization_results.json"))}
cache = json.load(open("/home/ebrahim/common/release_date_cache.json"))

TLC = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def seqof(ch): return "".join(TLC[r.name] for r in ch if r.name in TLC)
def mut(a): return "A" if a != "A" else "G"
def best_match_chain(st, target):
    best, br = None, -1
    for ch in st[0]:
        s = seqof(ch)
        if len(s) < 4: continue
        r = difflib.SequenceMatcher(None, s, target, autojunk=False).ratio()
        if r > br: br, best = r, ch
    return best

tc = [940]   # tid o940+ (free; max existing o932)
jobs = []
for pdb, aidxs in agidx.items():
    aidxs = [int(x) for x in aidxs]
    base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    pi = 0; cif2tid = {}; info = []
    for e in obj["sequences"]:
        if "proteinChain" not in e: continue
        if pi in aidxs:
            row = loc[(pdb, str(pi))]
            cif = os.path.join(PRED, row["plddt_pick_pose"])
            st = gemmi.read_structure(cif); st.setup_entities()
            match = best_match_chain(st, qseqs[pi]).name
            if cif not in cif2tid:
                tid = f"o{tc[0]:03d}"; tc[0] += 1
                if tid not in cache: cache[tid] = {"release_date": "2000-01-01"}
                open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))
                cif2tid[cif] = tid
            tid = cif2tid[cif]; q = qseqs[pi]; L = len(q); hs = list(q)
            for p in (L // 3, 2 * L // 3): hs[p] = mut(hs[p])
            a3mp = f"{A3M}/{pdb}_q{pi}.a3m"
            open(a3mp, "w").write(
                f">query\n{q}\n>{tid}_{match}/1-{L} [subseq from] mol:protein length:{L}  pddag\n{''.join(hs)}\n")
            e["proteinChain"]["templatesPath"] = a3mp
            info.append((pi, f"ANTIGEN tmpl {match} pl{row['plddt_pick_meanpLDDT']} ({row['plddt_pick_pose']})"))
        else:
            e["proteinChain"].pop("templatesPath", None)   # antibody: NO template (== free run state)
            info.append((pi, "ANTIBODY freed"))
        pi += 1
    obj["name"] = f"{pdb}_pddag"
    json.dump([obj], open(f"{TMPLDIR}/{pdb}-pddag.json", "w"))
    jobs.append({"pdb": pdb, "json": f"{TMPLDIR}/{pdb}-pddag.json", "seeds": list(range(101, 151))})
    print(f"{pdb}: {info}")

json.dump(cache, open("/home/ebrahim/common/release_date_cache.json", "w"))
json.dump(jobs, open(f"{EXP}/artifacts/pddag_jobs.json", "w"), indent=1)
print(f"\nBUILD_PDDAG_DONE: {len(jobs)} jobs -> {TMPLDIR}")
