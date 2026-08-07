#!/usr/bin/env python
"""Build `ifacelddt26` and `globallddt26`: the two lDDT template arms, PER INTERFACE, on the 26.

§4 ran these on the 15-decoy set with one job per PDB. Two things change here:

1. **Per interface, not per PDB.** For a chain in two interfaces (8yxi C is in both A-C and B-C) the
   old build defined "interface residues" as the UNION over its interfaces, so when A-C was scored,
   C's donor had been chosen partly by how well it modelled its contacts with B. Here the interface
   residues are those of the interface UNDER TEST, and each interface is its own run. Identical for the
   14 single-interface complexes; only the 6 two-interface PDBs differ.
2. **The canonical 26**, so the numbers sit alongside `dockqdonor`.

DONOR GRANULARITY: per chain, independently -- unlike DockQ, lDDT genuinely has a per-chain value, so
this is the criterion's native form and matches the published arms (mix-and-match donors).

  globallddt26  each chain <- the pose with the best WHOLE-CHAIN lDDT vs its native chain
  ifacelddt26   each chain <- the pose with the best lDDT over THIS interface's contact residues

Both are ORACLES: they need the native structure both to score accuracy and (for ifacelddt) to define
which residues count. Neither is deployable; they measure a ceiling.

Template mechanics identical to build_dockq_templates.py / README 6.3: donor pose registered as a
fake-old mmCIF under a fresh tid, 2-line a3m per chain, two residues mutated so Protenix reads it as a
template, multichain_mask so NO cross-chain information is carried.

Usage:  .../python scripts/build_lddt_arms.py
Writes: artifacts/sweep_tmpl/{ifacelddt26,globallddt26}/<interface>.json,
        artifacts/sweep_a3m/{ifacelddt26,globallddt26}_*.a3m,
        artifacts/lddt26_{ifacelddt26,globallddt26}_jobs.json, artifacts/lddt26_donors.json
"""
import json, os, sys, glob, difflib, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED, SEEDS, native_chains
from fold_quality import TLC, nat_res, model_chain

JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
A3M = f"{EXP}/artifacts/sweep_a3m"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
EXCLUDE = {"7z2m_AJ", "8udz_AC"}
os.makedirs(A3M, exist_ok=True)

sc = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
DECOYS = {k: v for k, v in sc.items()
          if v["free_status"] == "confident_decoy" and k not in EXCLUDE}
TBL = json.load(open(f"{EXP}/artifacts/fold_selection_table.json"))
cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
_n = [0]


def new_tid():
    while True:
        _n[0] += 1
        t = f"L{_n[0]:03d}"
        if t not in TAKEN:
            TAKEN.add(t); return t


def mut(a): return "A" if a != "A" else "G"


def native_seqs(pdb):
    """{label_asym_id: sequence} -- label ids, never gemmi's auth naming (see build_dockq_templates)."""
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


def iface_idx_pair(pdb, ch, partner):
    """0-based indices into nat_res(pdb,ch) contacting THIS partner only (not a union)."""
    D = native_chains(pdb)
    XA, IA = D[ch]; XB, _ = D[partner]
    order = {v: i for i, v in enumerate(sorted(set(IA), key=lambda x: int(x)))}
    m = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < 5.0
    return {order[IA[i]] for i in np.where(m.any(1))[0]}


def best_chain(st, q):
    best, br = None, -1
    for c in st[0]:
        s = "".join(TLC[r.name] for r in c if r.name in TLC)
        if not s: continue
        r_ = difflib.SequenceMatcher(None, s, q, autojunk=False).ratio()
        if r_ > br: br, best = r_, c
    return best


ARMS = ("globallddt26", "ifacelddt26")
jobs = {a: [] for a in ARMS}
donors, skipped = {}, []

for arm in ARMS:
    os.makedirs(f"{EXP}/artifacts/sweep_tmpl/{arm}", exist_ok=True)
    for lab in sorted(DECOYS):
        v = DECOYS[lab]; pdb = v["pdb"]
        parts = v["chains"].split("-")
        c1, c2 = (parts[0], "-".join(parts[1:])) if len(parts) > 2 else (parts[0], parts[1])
        if pdb not in TBL:
            skipped.append(f"{arm} {lab}: {pdb} absent from fold_selection_table"); continue

        base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
        obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
        qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
        natseq = native_seqs(pdb)

        cif2tid, pi, info, ok, picks = {}, 0, [], True, {}
        for e in obj["sequences"]:
            if "proteinChain" not in e:
                continue
            q = qseqs[pi]
            nc = max(natseq, key=lambda c: difflib.SequenceMatcher(None, natseq[c], q,
                                                                   autojunk=False).ratio())
            rec = TBL[pdb].get(str(pi))
            if rec is None or nc not in (c1, c2):
                # Not a chain of the interface under test -> no template, so the intervention is
                # confined to the tested interface's two chains (matches the published arms).
                e["proteinChain"].pop("templatesPath", None)
                info.append(f"q{pi}({nc}):free"); pi += 1; continue

            poses = rec["poses"]                    # [seed, sample, global_lddt, iface_lddt, gp, ip]
            if arm == "globallddt26":
                cand = [(p[0], p[1], p[2]) for p in poses if p[2] is not None]
            else:
                partner = c2 if nc == c1 else c1
                idx = iface_idx_pair(pdb, nc, partner)
                if not idx:
                    skipped.append(f"{arm} {lab} q{pi}({nc}): no contact residues"); ok = False; break
                # recompute the interface lDDT for THIS pair from the cached per-residue scores
                per = rec.get("per_residue_lddt")
                if per is None:
                    # table has only the union-based value; fall back to it and record that
                    cand = [(p[0], p[1], p[3]) for p in poses if p[3] is not None]
                    info.append(f"q{pi}({nc}):UNION-il")
                else:
                    cand = []
                    for p in poses:
                        d = per.get(f"{p[0]}_{p[1]}")
                        if not d: continue
                        vals = [d[str(i)] for i in idx if str(i) in d]
                        if vals: cand.append((p[0], p[1], float(np.mean(vals))))
            if not cand:
                skipped.append(f"{arm} {lab} q{pi}({nc}): no pose has the criterion"); ok = False; break
            s, k, val = max(cand, key=lambda x: x[2])

            cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
            st = gemmi.read_structure(cif); st.setup_entities()
            mc = best_chain(st, q)
            if mc is None:
                skipped.append(f"{arm} {lab} q{pi}: no model chain matched"); ok = False; break
            if cif not in cif2tid:
                tid = new_tid(); cache.setdefault(tid, {"release_date": "2000-01-01"})
                open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))
                cif2tid[cif] = tid
            tid = cif2tid[cif]
            L = len(q); hs = list(q)
            for pos in (L // 3, 2 * L // 3):
                hs[pos] = mut(hs[pos])
            a3mp = f"{A3M}/{arm}_{lab}_q{pi}.a3m"
            open(a3mp, "w").write(
                f">query\n{q}\n>{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  {arm}\n"
                f"{''.join(hs)}\n")
            e["proteinChain"]["templatesPath"] = a3mp
            picks[f"q{pi}"] = {"native_chain": nc, "seed": s, "sample": k, "criterion": round(val, 4)}
            info.append(f"q{pi}({nc}):s{s}/{k}={val:.3f}")
            pi += 1
        if not ok:
            continue
        obj["name"] = f"{pdb}_{c1}{c2}_{arm}".replace("-", "")
        jp = f"{EXP}/artifacts/sweep_tmpl/{arm}/{lab}.json"
        json.dump([obj], open(jp, "w"))
        jobs[arm].append({"run": arm, "pdb": pdb, "interface": lab, "json": jp,
                          "name": obj["name"], "seeds": list(SEEDS), "set": "decoy"})
        donors[f"{arm}|{lab}"] = picks
        print(f"{arm:14} {lab:12} {'  '.join(info)}", flush=True)

json.dump(cache, open(CACHE_F, "w"))
for arm in ARMS:
    p = f"{EXP}/artifacts/lddt26_{arm}_jobs.json"
    json.dump(jobs[arm], open(p, "w"), indent=1)
    assert len(jobs[arm]) == len({j["interface"] for j in jobs[arm]}), f"{arm}: duplicate interface job"
    print(f"BUILT {arm}: {len(jobs[arm])} interface jobs -> {p}")
json.dump(donors, open(f"{EXP}/artifacts/lddt26_donors.json", "w"), indent=1)
if skipped:
    print(f"SKIPPED {len(skipped)}:")
    for s in skipped[:30]:
        print("  " + s)
