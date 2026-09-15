#!/usr/bin/env python
"""On a seed that splits cleanly, do the OFF-EPITOPE poses all land on the SAME wrong surface?

THE QUESTION. 54% of high-DockQ-variance seeds split with no intermediate pose: some poses recover the
native epitope (recall > 0.5), the rest recover essentially none of it (< 0.1). Epitope RECALL cannot
say whether those off-epitope poses agree with each other -- two poses both scoring 0.00 might sit on
one alternative surface or on opposite faces of the antigen. That needs the residue SETS.

  epitope of a pose   antigen residues with a heavy atom within 5.0 A of any antibody heavy atom
                      (contact_residues reused verbatim from scripts/epitope_clusters.py -- the same
                       CAPRI definition, cutoff and OST selection used everywhere else here)
  agreement           Jaccard of two poses' epitope sets. Reported WITHIN a seed, among its
                      off-epitope poses only.

CHAIN MAPPING comes from each pose's own PXMeter json (`ref_to_model_chain_mapping`), not from a
sequence matcher and not from the reference letters -- model chain names differ from reference names,
and on multi-copy assemblies the mapping changes per pose. epitope_clusters.py takes this from OST
jsons, which do not exist for the PXM-2025 pool; this is the same correspondence from the scorer that
was actually used.

Usage: offepitope_cluster.py [--jobs 24] [--spread 0.4]
"""
import argparse
import collections
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
CUTOFF = 5.0
HI, LO = 0.5, 0.1          # "on the epitope" / "off the epitope", as used in the split analysis


def _ost():
    import ost
    from ost import io, mol
    return ost, io, mol


def contact_residues(ent, ab_chain, ag_chain, cutoff=CUTOFF):
    """Antigen residue numbers within `cutoff` of the antibody chain (heavy atoms, no hydrogens).
    Verbatim from scripts/epitope_clusters.py so results stay comparable to the FoldBench run."""
    _ost()
    sel_ab = ent.Select(f"chain='{ab_chain}' and ele!=H")
    sel_ag = ent.Select(f"chain='{ag_chain}' and ele!=H")
    if not sel_ab.GetAtomCount() or not sel_ag.GetAtomCount():
        return set()
    ab = np.array([[a.GetPos()[0], a.GetPos()[1], a.GetPos()[2]] for a in sel_ab.atoms])
    out = set()
    for r in sel_ag.residues:
        pos = np.array([[a.GetPos()[0], a.GetPos()[1], a.GetPos()[2]] for a in r.atoms])
        if pos.size and np.min(np.linalg.norm(pos[:, None, :] - ab[None, :, :], axis=-1)) <= cutoff:
            out.add(r.GetNumber().GetNum())
    return out


def jac(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pose_epitope(arg):
    """-> (key, epitope set) for one pose, or (key, None) if it cannot be resolved."""
    pdb, seed, smp, c_ab, c_ag = arg
    cif = f"{P}/out/{pdb}/{pdb}/seed_{seed}/predictions/{pdb}_sample_{smp}.cif"
    if not os.path.exists(cif):
        return (pdb, seed, smp), None
    try:
        _, io, _ = _ost()
        ent = io.LoadMMCIF(cif)
        return (pdb, seed, smp), contact_residues(ent, c_ab, c_ag)
    except Exception:
        return (pdb, seed, smp), None


def model_chains(pdb, seed, smp, c1, c2):
    """Per-pose ref->model chain pair from the pose's own PXMeter json."""
    try:
        m = json.load(open(f"{P}/pxm_json/{pdb}_{seed}_{smp}.json")).get("ref_to_model_chain_mapping") or {}
        return m[c1], m[c2]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--spread", type=float, default=0.4)
    ap.add_argument("--out", default=f"{EXP}/artifacts/offepitope_clusters.csv")
    A = ap.parse_args()

    # recall per pose, and which chain is the antigen (the epitope file records the roles)
    rec, role = {}, {}
    for r in csv.DictReader(open(f"{P}/epitope_overlap_pxm.csv")):
        key = (r["pdb"], frozenset((r["ab_chain"], r["ag_chain"])))
        rec[(r["pdb"], frozenset((r["ab_chain"], r["ag_chain"])), r["seed"], r["sample"])] = float(r["recall"])
        role[key] = (r["ab_chain"], r["ag_chain"])

    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        try:
            pool[(r["pdb"], r["chain1"], r["chain2"])][r["seed"]].append((r["sample"], float(r["dockq"])))
        except (TypeError, ValueError):
            continue

    # the clean-split seeds: >=1 pose on the epitope, >=1 off, and NO pose in between
    work, seedinfo = [], []
    for key, S in pool.items():
        fs = frozenset((key[1], key[2]))
        if (key[0], fs) not in role:      # role is keyed (pdb, {chains}) -- both parts, not just fs
            continue
        ab, ag = role[(key[0], fs)]
        for seed, v in S.items():
            if len(v) < 5:
                continue
            rs = [rec.get((key[0], fs, seed, s)) for s, _ in v]
            if any(x is None for x in rs):
                continue
            dq = [d for _, d in v]
            if max(dq) - min(dq) <= A.spread:
                continue
            hi = [s for (s, _), x in zip(v, rs) if x > HI]
            lo = [s for (s, _), x in zip(v, rs) if x < LO]
            if not hi or not lo or len(hi) + len(lo) != 5:      # clean split only
                continue
            iface = f"{key[0]}_{key[1]}{key[2]}"
            seedinfo.append((iface, key[0], seed, fs, hi, lo))
            for s in hi + lo:
                mc = model_chains(key[0], seed, s, ab, ag)
                if mc:
                    work.append((key[0], seed, s, mc[0], mc[1]))
    print(f"  clean-split seeds: {len(seedinfo)}   poses to extract: {len(work)}", flush=True)

    with Pool(A.jobs) as pl:
        got = dict(pl.map(pose_epitope, work, chunksize=8))
    ok = sum(1 for v in got.values() if v)
    print(f"  epitopes extracted: {ok}/{len(work)}", flush=True)

    rows = []
    for iface, pdb, seed, fs, hi, lo in seedinfo:
        offs = [got.get((pdb, seed, s)) for s in lo]
        ons = [got.get((pdb, seed, s)) for s in hi]
        offs = [x for x in offs if x]
        ons = [x for x in ons if x]
        if len(offs) < 2:                 # need >=2 off-epitope poses to ask whether they agree
            continue
        pj = [jac(offs[i], offs[j]) for i in range(len(offs)) for j in range(i + 1, len(offs))]
        cross = [jac(o, n) for o in offs for n in ons] if ons else []
        rows.append(dict(iface=iface, seed=seed, n_off=len(offs), n_on=len(ons),
                         off_off=sum(pj) / len(pj),
                         off_off_min=min(pj),
                         off_on=(sum(cross) / len(cross)) if cross else float("nan"),
                         same_surface=int(min(pj) >= 0.5)))
    with open(A.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {A.out}  ({len(rows)} seeds with >=2 off-epitope poses)\n")

    import statistics as st
    oo = [r["off_off"] for r in rows]
    same = sum(r["same_surface"] for r in rows)
    print(f"  {'':<44}{'mean':>8}{'median':>9}")
    print(f"  {'Jaccard BETWEEN the off-epitope poses':<44}{st.mean(oo):>8.3f}{st.median(oo):>9.3f}")
    on = [r["off_on"] for r in rows if r["off_on"] == r["off_on"]]
    if on:
        print(f"  {'Jaccard off-epitope vs on-epitope (control)':<44}{st.mean(on):>8.3f}{st.median(on):>9.3f}")
    print()
    print(f"  all off-epitope poses on ONE surface (min pairwise Jaccard >= 0.5): "
          f"{same}/{len(rows)} ({same/len(rows)*100:.0f}%)")
    for lo_, hi_ in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
        n = sum(1 for x in oo if lo_ <= x < hi_)
        print(f"    off-off Jaccard {lo_:.1f}-{hi_:.1f}: {n:>4}  ({n/len(oo)*100:>4.1f}%)")


if __name__ == "__main__":
    main()
