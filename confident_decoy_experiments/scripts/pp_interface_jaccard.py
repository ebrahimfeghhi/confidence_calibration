#!/usr/bin/env python
"""Interface residue-set overlap for protein-protein poses, the analogue of epitope/paratope jaccard.

There is no antibody here, so the two sides are just the two chains of the interface. For each pose
and each side we take the residues within 5 A of the partner (same definition used throughout), map
them into ground-truth numbering, and report:

  jaccard_to_native  -- did this pose contact the right residues on that side
  pairwise agreement -- do the poses of one interface agree WITH EACH OTHER (the quantity that
                        decides whether there is a single coherent wrong answer to repel from,
                        which LRMSD spread only proxies)

Chains come from OST's dockq_interfaces row for THAT pose (ref1, ref2, mdl1, mdl2), so model and
reference sides can never be mismatched.
"""
import csv, glob, itertools, json, os, re, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "scripts")
from cdr_h3_rmsd import _ost, align_map, chain_residues
from epitope_clusters import contact_residues

GT = "artifacts/pp/gt"
OSTD = "artifacts/pp/ost"


def ost_row(pdb, seed, smp, c1, c2):
    try:
        d = json.load(open(f"{OSTD}/{pdb}_{seed}_{smp}.json"))
    except Exception:
        return None
    base = lambda c: re.sub(r"-\d+$", "", c)
    for row in d.get("dockq_interfaces") or []:
        if len(row) < 4:
            continue
        if {base(row[0]), base(row[1])} == {c1, c2}:
            return (row[0], row[1], row[2], row[3]) if base(row[0]) == c1 else \
                   (row[1], row[0], row[3], row[2])
    return None


def one(arg):
    pdb, c1, c2, cif, seed, smp = arg
    try:
        _, io, _ = _ost()
        r = ost_row(pdb, seed, smp, c1, c2)
        if not r:
            return None
        r1, r2, m1, m2 = r
        mdl = io.LoadMMCIF(cif, fault_tolerant=True)
        ref = io.LoadMMCIF(f"{GT}/{pdb}-assembly1.cif", fault_tolerant=True)
        out = []
        for mc, rc, other_m, other_r in ((m1, r1, m2, r2), (m2, r2, m1, r1)):
            mres, rres = chain_residues(mdl, mc), chain_residues(ref, rc)
            amap = align_map("".join(x[1] for x in mres), "".join(x[1] for x in rres))
            m2g = {mres[i][0]: rres[j][0] for i, j in amap.items()}
            pred = {m2g[x] for x in contact_residues(mdl, other_m, mc) if x in m2g}
            nat = contact_residues(ref, other_r, rc)
            u = pred | nat
            out.append((sorted(pred), len(pred & nat) / len(u) if u else 0.0, len(nat)))
        return (seed, smp, out[0], out[1])
    except Exception:
        return None


def main():
    P = list(csv.DictReader(open("artifacts/pp/dockq_pp.csv")))
    import collections
    by = collections.defaultdict(list)
    for r in P:
        pi = r["pair_iptm"]
        if pi in ("", "None"):
            continue
        by[(r["pdb"], r["chain1"], r["chain2"])].append((float(pi), float(r["dockq"])))
    fail = [k for k, v in by.items() if len(v) >= 100
            and v[int(np.argmax([x[0] for x in v]))][1] < 0.23]
    print(f"{len(fail)} protein-protein interfaces with selected DockQ < 0.23\n", flush=True)
    res = {}
    for pdb, c1, c2 in fail:
        cifs = sorted(glob.glob(f"artifacts/pp/out/{pdb}/**/seed_*/predictions/*_sample_*.cif",
                                recursive=True))
        work = []
        for f in cifs:
            m = re.search(r"seed_(\d+)/predictions/.*_sample_(\d+)\.cif$", f)
            if m:
                work.append((pdb, c1, c2, f, m.group(1), m.group(2)))
        if len(work) < 50:
            continue
        with Pool(24) as p:
            r = [x for x in p.map(one, work, chunksize=6) if x]
        if len(r) < 50:
            continue
        res[f"{pdb}_{c1}{c2}"] = r
        print(f"  {pdb}_{c1}{c2}: {len(r)} poses", flush=True)
    np.save("artifacts/pp_iface_jaccard.npy", res, allow_pickle=True)
    print(f"\nsaved {len(res)} interfaces -> artifacts/pp_iface_jaccard.npy")


if __name__ == "__main__":
    main()
