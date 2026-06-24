"""Cluster the antigen docking poses across all seeds for 7txw A|C.

Do the 50 seeds converge on a few common antigen orientations (a dominant
"decoy" pose + the native pose), or are they scattered? Superpose every seed's
antibody (chain A) onto a common reference, carry its antigen (chain C) into that
frame, then cluster seeds by pairwise antigen-CA RMSD (connected components at a
distance threshold).
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_7txw_poses import parse_cif_ca, kabsch, common, PRED  # noqa: E402

GOOD = {10, 14, 19, 33}
DOCKQ_CSV = Path("/home/ebrahim/tts_abag/results/csv_files/"
                 "dockq_ost_esmfold2fast_tts_all_samples.csv")
THRESHOLDS = [5.0, 10.0]   # A; poses within this antigen-RMSD = same orientation


def best_dockq_per_seed():
    out = {}
    with open(DOCKQ_CSV) as f:
        for r in csv.DictReader(f):
            if r["assembly_id"] != "7txw-assembly1":
                continue
            if {r["chain1"], r["chain2"]} != {"A", "C"}:
                continue
            try:
                dq = float(r["dockq"])
            except ValueError:
                continue
            out[r["seed"]] = max(out.get(r["seed"], 0.0), dq)
    return out


def connected_components(D, thr):
    n = len(D)
    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if not seen[v] and D[u, v] < thr:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def main():
    dq = best_dockq_per_seed()
    seed_dirs = sorted([d.name for d in PRED.iterdir() if d.name.startswith("seed_")],
                       key=lambda s: int(s.split("_")[1]))

    ref = parse_cif_ca(PRED / "seed_10" / "predictions" / "7txw_sample_0.cif")
    antigen_keys = sorted(ref["C"])

    labels, coords, dqs = [], [], []
    for sd in seed_dirs:
        f = PRED / sd / "predictions" / "7txw_sample_0.cif"
        if not f.exists():
            continue
        st = parse_cif_ca(f)
        P, Q, _ = common(st, ref, "A")
        R, t = kabsch(P, Q)
        keys = [k for k in antigen_keys if k in st["C"]]
        ag = np.array([R @ st["C"][k] + t for k in keys])
        labels.append(sd.replace("seed_", "s"))
        coords.append(ag)
        dqs.append(dq.get(sd.split("_")[1], float("nan")))

    n = len(labels)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.sqrt(((coords[i] - coords[j]) ** 2).sum(1).mean()))
            D[i, j] = D[j, i] = r

    is_good = [int(lbl[1:]) in GOOD for lbl in labels]
    print(f"{n} seeds (sample_0).  antigen-pose RMSD after antibody superposition.\n")
    for thr in THRESHOLDS:
        comps = connected_components(D, thr)
        comps.sort(key=len, reverse=True)
        print(f"=== clusters at < {thr:.0f} A ===  ({len(comps)} clusters)")
        for ci, comp in enumerate(comps):
            good_in = [labels[i] for i in comp if is_good[i]]
            cdqs = [dqs[i] for i in comp if dqs[i] == dqs[i]]
            mean_dq = np.mean(cdqs) if cdqs else float("nan")
            tag = f"  <-- contains GOOD {good_in}" if good_in else ""
            members = " ".join(labels[i] for i in comp)
            print(f"  cluster {ci}: n={len(comp):2d}  meanDockQ={mean_dq:.2f}{tag}")
            print(f"            {members}")
        print()


if __name__ == "__main__":
    main()
