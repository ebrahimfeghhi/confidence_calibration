#!/usr/bin/env python
"""Per-SAMPLE epitope Jaccard at one recycle, instead of the best-of-5 that traj_analyze stores.

WHY: traj_analyze reports max over the 5 diffusion samples, so "seed 116 = 1.00" is ambiguous --
it could be 5/5 samples on the crystal epitope or 1/5. That distinction decides whether the trunk
has committed to the site (all samples agree) or the site is one option among five (selection
problem, not a trunk problem).

Usage: .../python scripts/traj_sample_detail.py --pdb 7yxu --iface 7yxu_CA --recycle 9
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "scripts")
from cdr_h3_rmsd import GT_DIR, _ost, align_map, chain_residues
from epitope_rmsd import native_epitope
from iface_chains import first_pose_with
from traj_analyze import chain_token_base, epitope_from_coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--iface", required=True)
    ap.add_argument("--recycle", type=int, default=9)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--offsite", action="store_true")
    A = ap.parse_args()

    d = A.dir or f"artifacts/traj_cycles/{A.pdb}"
    seeds = ([int(x) for x in A.seeds.split(",")] if A.seeds
             else json.load(open("artifacts/traj20_seeds.json"))[A.pdb]["seeds"])
    per = {}
    for f in sorted(glob.glob(f"{d}/*_coords.npz")):
        m = re.search(r"_pass(\d+)_cycle(\d+)_coords", os.path.basename(f))
        per.setdefault(int(m.group(1)), {})[int(m.group(2))] = f

    ab_l, ag_l = A.iface.rsplit("_", 1)[1][0], A.iface.rsplit("_", 1)[1][1]
    ch, _, _ = first_pose_with(A.pdb, ab_l, ag_l)
    ci = lambda c: ord(c.split("-")[0][0].upper()) - 65
    ag_asym, ab_asyms = ci(ch["mdl_ag"]), [ci(ch["mdl_ab"])]
    _, io, _ = _ost()
    ref = io.LoadMMCIF(f"{GT_DIR}/{A.pdb}-assembly1.cif", fault_tolerant=True)
    nat = native_epitope(ref, ch["ref_ab"], ch["ref_ag"])
    rres = chain_residues(ref, ch["ref_ag"])
    _b = json.load(open(f"/home/ebrahim/Protenix/foldbench_abag/jsons/{A.pdb}-final-updated.json"))
    _b = _b[0] if isinstance(_b, list) else _b
    _seqs = [v["sequence"] for x in _b["sequences"] for t, v in x.items()
             if t == "proteinChain" for _ in range(int(v.get("count", 1)))]
    amap = align_map(_seqs[ag_asym], "".join(x[1] for x in rres))
    J = lambda a, b: len(a & b) / len(a | b) if (a | b) else 1.0

    print(f"{A.iface} recycle {A.recycle}: Jaccard to crystal epitope, one column per diffusion sample")
    print(f"{'seed':>6}  " + "".join(f"{'s'+str(i):>7}" for i in range(5)) + f"{'n>=0.5':>9}")
    on, off, labels = [], [], []
    for p in sorted(per):
        f = per[p].get(A.recycle)
        if not f:
            continue
        base, _ = chain_token_base(f, ag_asym)
        gt = lambda s: {rres[amap[x - base]][0] for x in s if (x - base) in amap}
        eps = epitope_from_coords(f, ag_asym, ab_asyms)
        vals = [J(gt(e), nat) for e in eps]
        s = seeds[p] if p < len(seeds) else p
        print(f"{s:>6}  " + "".join(f"{v:>7.2f}" for v in vals)
              + f"{sum(v >= 0.5 for v in vals):>9}")
        for i, (v, e) in enumerate(zip(vals, eps)):
            (on if v >= 0.5 else off).append(gt(e))
            if v < 0.5:
                labels.append(f"{s}.s{i}")

    if not A.offsite:
        return
    # Do the OFF-epitope samples agree with each other, or scatter? One shared decoy site would
    # show up as high pairwise Jaccard among them; scatter means there is no single competing basin.
    def pw(sets):
        return [J(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    print(f"\noff-epitope samples (Jaccard to crystal < 0.5): n={len(off)}")
    o = pw(off)
    print(f"  mean pairwise Jaccard among themselves: {np.mean(o):.2f}  "
          f"(median {np.median(o):.2f}, frac > 0.5: {np.mean(np.array(o) > 0.5):.2f})")
    if on:
        print(f"  on-epitope samples n={len(on)}, mean pairwise among themselves: {np.mean(pw(on)):.2f}")
    M = np.array([[J(a, b) for b in off] for a in off])
    print("\n  pairwise Jaccard matrix (rows/cols = seed.sample):")
    print("      " + "".join(f"{l.split('.')[0][-3:]+l.split('.')[1]:>8}" for l in labels))
    for l, row in zip(labels, M):
        print(f"  {l:>8}" + "".join(f"{v:>8.2f}" for v in row))
    print(f"\n  empty contact sets (no antigen contact at all): "
          f"{sum(1 for e in off if not e)}/{len(off)}")


if __name__ == "__main__":
    main()
