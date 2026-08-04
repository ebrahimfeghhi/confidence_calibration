#!/usr/bin/env python
"""Build `topiptm26`: template from the folds of the pose the model was MOST CONFIDENT about.

The top-interface-pair_iptm pose IS the confident decoy (DockQ < 0.23 by the set's definition). So
this feeds the model back the folds of the pose it already wrongly believes in. Two opposed
predictions, both plausible:

  A) LEVEL THE FIELD  -- if the decoy's high ipTM comes partly from its fold being uniquely
     self-consistent, forcing every seed onto that fold removes the fold advantage, and seeds then
     compete on docking alone -> the decoy's edge shrinks.
  B) REINFORCE THE DECOY -- if the top-ipTM pose's fold is specifically shaped to fit the WRONG
     epitope, templating it makes the wrong dock more geometrically satisfying -> decoy wins harder.

This is the exact mirror of `dockqdonor` (which templates from the best-DOCKED pose): same complexes,
same machinery, same 250 poses, donor chosen by the opposite criterion. Together they separate "any
good fold helps" from "the fold of a correctly-docked pose helps".

WHY IT IS BEING RUN AGAIN: the 2026-07-28 `topiptm` arm reported 0/5 and was read as evidence for (B).
It is VOID -- 451 chain-loads, every one logging "Found 0 templates"; its templates never loaded, so it
measured an untemplated run. Same for `topdockq` (1601) and `sposefv` (1205). See RESULTS.md banner.

Design matches build_dockq_templates.py exactly so the two arms are comparable:
  - ONE donor pose per INTERFACE, chosen by max interface pair_iptm over the 250 free-run poses
  - EVERY protein chain templated from that one pose (incl. chains outside the tested interface)
  - templates carry intra-chain geometry only (multichain_mask), so no docking is transferred
  - native chain ids via label_asym_id, never gemmi auth naming

Usage:  .../python scripts/build_topiptm_templates.py
Writes: artifacts/sweep_tmpl/topiptm26/<interface>.json, artifacts/sweep_a3m/topiptm26_*.a3m,
        artifacts/topiptm26_jobs.json, artifacts/topiptm26_donors.json
"""
import csv, json, os, sys, glob, difflib, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED, SEEDS, SAMPLES, THRESH

JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPL = f"{EXP}/artifacts/sweep_tmpl/topiptm26"; A3M = f"{EXP}/artifacts/sweep_a3m"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
CACHES = [f"{EXP}/artifacts/abag_all_ost", f"{EXP}/artifacts/decoy_set_ost"]
EXCLUDE = {"7z2m_AJ", "8udz_AC"}
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

sc = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
DECOYS = {k: v for k, v in sc.items()
          if v["free_status"] == "confident_decoy" and k not in EXCLUDE}
cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
_n = [0]


def new_tid():
    while True:
        _n[0] += 1
        t = f"i{_n[0]:03d}"
        if t not in TAKEN:
            TAKEN.add(t); return t


def mut(a): return "A" if a != "A" else "G"


def native_seqs(pdb):
    """{label_asym_id: sequence} -- label ids (gemmi's auth naming differs for 8bf4/8che/8fbw/8slb/8the)."""
    path = f"/home/ebrahim/FoldBench/examples/ground_truths/{pdb}-assembly1.cif"
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_comp_id", "label_seq_id", "group_PDB")]
    seen, out = collections.defaultdict(set), collections.defaultdict(list)
    for a, comp, sid, g in zip(*cols):
        if g != "ATOM" or comp not in TLC or sid in seen[a]:
            continue
        seen[a].add(sid); out[a].append(TLC[comp])
    return {k: "".join(v) for k, v in out.items() if len(v) >= 4}


def best_chain(st, q):
    best, br = None, -1
    for c in st[0]:
        s = "".join(TLC[r.name] for r in c if r.name in TLC)
        if not s: continue
        r_ = difflib.SequenceMatcher(None, s, q, autojunk=False).ratio()
        if r_ > br: br, best = r_, c
    return best


def poses_for(pdb, c1, c2):
    """{(seed, sample): (pair_iptm, dockq)} for one labelled interface, from the cached OST runs."""
    out = {}
    for s in SEEDS:
        for k in SAMPLES:
            for cdir in CACHES:
                o = f"{cdir}/{pdb}_{s}_{k}.json"
                if not (os.path.exists(o) and os.path.getsize(o)):
                    continue
                try:
                    d = json.load(open(o))
                    M = json.load(open(f"{PRED}/{pdb}/seed_{s}/predictions/"
                                       f"{pdb}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
                except Exception:
                    continue
                qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
                for i, t in enumerate(ifs):
                    if {t[0], t[1]} == {c1, c2} and i < len(qs):     # BY NAME
                        x, y = ord(t[2]) - 65, ord(t[3]) - 65
                        if 0 <= x < len(M) and 0 <= y < len(M):
                            out[(s, k)] = (M[x][y], qs[i])
                        break
                if (s, k) in out:
                    break
    return out


jobs, donors, skipped = [], {}, []
for lab in sorted(DECOYS):
    v = DECOYS[lab]; pdb = v["pdb"]
    parts = v["chains"].split("-")
    c1, c2 = (parts[0], "-".join(parts[1:])) if len(parts) > 2 else (parts[0], parts[1])

    P = poses_for(pdb, c1, c2)
    if not P:
        skipped.append(f"{lab}: no cached poses"); continue
    (s, k) = max(P, key=lambda x: P[x][0])          # highest interface pair_iptm = the decoy itself
    iptm, dq = P[(s, k)]
    if dq >= THRESH:
        skipped.append(f"{lab}: top-ipTM pose has DockQ {dq:.3f} >= {THRESH} -- not a decoy, skipped")
        continue

    cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
    if not os.path.exists(cif):
        skipped.append(f"{lab}: donor cif missing"); continue

    base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    natseq = native_seqs(pdb)
    st = gemmi.read_structure(cif); st.setup_entities()

    tid = new_tid(); cache.setdefault(tid, {"release_date": "2000-01-01"})
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))

    pi, info, ok, chains = 0, [], True, {}
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = qseqs[pi]
        nc = max(natseq, key=lambda c: difflib.SequenceMatcher(None, natseq[c], q,
                                                               autojunk=False).ratio())
        mc = best_chain(st, q)
        if mc is None:
            skipped.append(f"{lab} q{pi}: no model chain matched"); ok = False; break
        L = len(q); hs = list(q)
        for pos in (L // 3, 2 * L // 3):
            hs[pos] = mut(hs[pos])
        a3mp = f"{A3M}/topiptm26_{lab}_q{pi}.a3m"
        open(a3mp, "w").write(
            f">query\n{q}\n>{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  topiptm26\n"
            f"{''.join(hs)}\n")
        e["proteinChain"]["templatesPath"] = a3mp
        chains[f"q{pi}"] = nc
        info.append(f"q{pi}({nc})<-{mc.name}")
        pi += 1
    if not ok:
        continue

    donors[lab] = {"pdb": pdb, "interface": [c1, c2], "seed": s, "sample": k,
                   "donor_pair_iptm": round(iptm, 3), "donor_dockq": round(dq, 3),
                   "n_chains_templated": len(chains), "chains": chains, "tid": tid}
    obj["name"] = f"{pdb}_{c1}{c2}_topiptm26".replace("-", "")
    jp = f"{TMPL}/{lab}.json"
    json.dump([obj], open(jp, "w"))
    jobs.append({"run": "topiptm26", "pdb": pdb, "interface": lab, "json": jp,
                 "name": obj["name"], "seeds": list(SEEDS), "set": "decoy"})
    print(f"{lab:12} donor seed{s}/s{k} ipTM={iptm:.3f} DockQ={dq:.3f}  chains: {' '.join(info)}",
          flush=True)

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{EXP}/artifacts/topiptm26_jobs.json", "w"), indent=1)
json.dump(donors, open(f"{EXP}/artifacts/topiptm26_donors.json", "w"), indent=1)
assert len(jobs) == len({j["interface"] for j in jobs}), "duplicate job per interface"
print(f"\nBUILT {len(jobs)} interface jobs (of {len(DECOYS)} decoys) -> artifacts/topiptm26_jobs.json")
if skipped:
    print(f"SKIPPED {len(skipped)}:")
    for s_ in skipped:
        print("  " + s_)
