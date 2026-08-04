#!/usr/bin/env python
"""Rebuild agonly / abonly / pddag templates for the 2 PDBs hit by the antigen-label bug
(8yxi, 8fbw) using the CORRECTED artifacts/agonly_antigen_idx.json.

  agonly = antigen chain keeps the NATIVE-fold template; antibody chains freed (templatesPath removed)
  abonly = antibody chains keep the NATIVE-fold template; antigen freed
  pddag  = antigen chain gets the best-pLDDT PREDICTED fold; antibody freed

Native a3m templates are copied from the ephemeral /tmp scratch into
artifacts/native_a3m/ and the JSONs point at the durable copies.
Run with Protenix venv python (gemmi needed for pddag chain matching).
"""
import json, os, sys, shutil, difflib
sys.path.insert(0, "/home/ebrahim/confident_decoy_experiments/scripts")
from cifconv import convert
import gemmi

EXP  = "/home/ebrahim/confident_decoy_experiments"
OLDT = "/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/tmpl/all"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
NA3M = f"{EXP}/artifacts/native_a3m"; os.makedirs(NA3M, exist_ok=True)
PDBS = ["8yxi", "8fbw"]
agidx = json.load(open(f"{EXP}/artifacts/agonly_antigen_idx.json"))
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

jobs = {"agonly": [], "abonly": [], "pddag": []}
tc = [960]   # fresh tid range for the rebuilt pddag templates
for pdb in PDBS:
    aidx = [int(x) for x in agidx[pdb]]
    nat = json.load(open(f"{OLDT}/{pdb}-nativetmpl.json"))
    natobj = nat[0] if isinstance(nat, list) else nat
    # copy native a3m to durable storage
    pi = 0
    for e in natobj["sequences"]:
        if "proteinChain" not in e: continue
        tp = e["proteinChain"].get("templatesPath")
        if tp and os.path.exists(tp):
            dst = f"{NA3M}/{os.path.basename(tp)}"
            if not os.path.exists(dst): shutil.copy(tp, dst)
            e["proteinChain"]["templatesPath"] = dst
        pi += 1

    for run in ("agonly", "abonly", "pddag"):
        obj = json.loads(json.dumps(natobj))
        qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
        pi = 0; info = []
        for e in obj["sequences"]:
            if "proteinChain" not in e: continue
            is_ag = pi in aidx
            keep_native = (is_ag and run == "agonly") or ((not is_ag) and run == "abonly")
            if run == "pddag":
                if is_ag:
                    row = loc.get((pdb, str(pi)))
                    if row is not None:
                        cif = os.path.join(PRED, row["plddt_pick_pose"])
                    else:
                        # antigen absent from the (pre-fix) localization table -> pick best mean-pLDDT now
                        import glob, statistics
                        best = (-1, None)
                        for s in range(101, 151):
                            for c in sorted(glob.glob(f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_*.cif")):
                                try: st = gemmi.read_structure(c); st.setup_entities()
                                except Exception: continue
                                ch = best_match_chain(st, qseqs[pi])
                                if ch is None: continue
                                pl = statistics.mean([at.b_iso for r in ch for at in r]) / 100.0
                                if pl > best[0]: best = (pl, c)
                        cif = best[1]
                        print(f"   [{pdb} q{pi}] computed pLDDT pick inline: {os.path.relpath(cif,PRED)} (pLDDT {best[0]:.3f})")
                    st = gemmi.read_structure(cif); st.setup_entities()
                    match = best_match_chain(st, qseqs[pi]).name
                    tid = f"o{tc[0]:03d}"; tc[0] += 1
                    if tid not in cache: cache[tid] = {"release_date": "2000-01-01"}
                    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))
                    q = qseqs[pi]; L = len(q); hs = list(q)
                    for p in (L//3, 2*L//3): hs[p] = mut(hs[p])
                    a3mp = f"{EXP}/artifacts/pddag_a3m/{pdb}_q{pi}.a3m"
                    open(a3mp, "w").write(
                        f">query\n{q}\n>{tid}_{match}/1-{L} [subseq from] mol:protein length:{L}  pddag\n{''.join(hs)}\n")
                    e["proteinChain"]["templatesPath"] = a3mp
                    info.append((pi, f"ANTIGEN pLDDT-tmpl {match}"))
                else:
                    e["proteinChain"].pop("templatesPath", None); info.append((pi, "ANTIBODY freed"))
            elif keep_native:
                info.append((pi, f"{'ANTIGEN' if is_ag else 'ANTIBODY'} NATIVE tmpl"))
            else:
                e["proteinChain"].pop("templatesPath", None)
                info.append((pi, f"{'ANTIGEN' if is_ag else 'ANTIBODY'} freed"))
            pi += 1
        obj["name"] = f"{pdb}_{run}"
        d = f"{EXP}/artifacts/relabel_tmpl"; os.makedirs(d, exist_ok=True)
        path = f"{d}/{pdb}-{run}.json"
        json.dump([obj], open(path, "w"))
        jobs[run].append({"pdb": pdb, "json": path, "seeds": list(range(101, 151))})
        print(f"{pdb:5} {run:7}: {info}")

json.dump(cache, open("/home/ebrahim/common/release_date_cache.json", "w"))
json.dump(jobs, open(f"{EXP}/artifacts/relabel_jobs.json", "w"), indent=1)
print(f"\nBUILD_RELABEL_DONE -> {EXP}/artifacts/relabel_tmpl ({sum(len(v) for v in jobs.values())} jobs)")
