#!/usr/bin/env python
"""How much does the ACTUAL epitope change from one recycle to the next?

Reads the coordinates decoded at every recycle (DUMP_CYCLE_DECODE=1: a full 200-step diffusion
rollout per cycle, observer only -- nothing is fed back, so the trajectory is a normal run).

Epitope of a cycle = antigen TOKENS with any atom within 5 A of any antibody atom, the same 5 A
heavy-atom definition used everywhere else in this project. No distogram, no proxy: this is the
epitope of the structure the model would actually have produced had it stopped at that cycle.

Change between cycles is reported as Jaccard of those residue sets, which needs no distance metric
and no threshold -- the two things that made the earlier distogram-based attempts hard to interpret.
"""
import glob, os, re, sys
import numpy as np

CUT = 5.0
AG_ASYM, AB_ASYM = 0, (1, 2)


def epitope(f):
    z = np.load(f)
    xyz, a2t, asym = z["coords"], z["atom_to_token_idx"], z["asym_id"]
    tok_asym = np.zeros(a2t.max() + 1, dtype=int)
    tok_asym[a2t] = asym[a2t] if asym.shape[0] == a2t.shape[0] else asym[:tok_asym.shape[0]][a2t]
    at_asym = asym[a2t] if asym.shape[0] > a2t.max() else tok_asym[a2t]
    ag = np.isin(at_asym, [AG_ASYM]); ab = np.isin(at_asym, list(AB_ASYM))
    A, B = xyz[ag], xyz[ab]
    tok_ag = a2t[ag]
    from scipy.spatial import cKDTree
    idx = cKDTree(B).query_ball_point(A, CUT)
    hit = np.array([len(i) > 0 for i in idx])
    return set(np.unique(tok_ag[hit]).tolist())


def main():
    order = {}
    for d in sorted(glob.glob("artifacts/decode_cycles/g*")):
        seeds = open(f"{d}/seed_order.txt").read().strip().split(",")
        for p, s in enumerate(seeds):
            order[(d, p)] = int(s)
    per = {}
    for d in sorted(glob.glob("artifacts/decode_cycles/g*")):
        for f in sorted(glob.glob(f"{d}/*_coords.npz")):
            m = re.search(r"_pass(\d+)_cycle(\d+)_coords", os.path.basename(f))
            seed = order[(d, int(m.group(1)))]
            per.setdefault(seed, {})[int(m.group(2))] = epitope(f)
    J = lambda a, b: len(a & b) / len(a | b) if (a | b) else 1.0
    seeds = sorted(per)
    print(f"8slb, {len(seeds)} stock seeds. Epitope = antigen residues within 5 A of the antibody in")
    print("the structure decoded AT THAT RECYCLE. Jaccard between consecutive recycles:")
    print("1.00 = identical epitope, 0.00 = no shared residues.\n")
    print(f"{'seed':<6}" + "".join(f"{k}->{k+1:<5}" for k in range(9)) + f"{'size@9':>9}")
    rows = []
    for s in seeds:
        c = per[s]
        r = [J(c[k], c[k + 1]) for k in range(9)]
        rows.append(r)
        print(f"{s:<6}" + "".join(f"{v:<7.2f}" for v in r) + f"{len(c[9]):>9}")
    r = np.array(rows)
    print(f"\n{'mean':<6}" + "".join(f"{v:<7.2f}" for v in r.mean(0)))
    print(f"\nJaccard of each cycle's epitope to that seed's FINAL epitope:")
    print(f"{'seed':<6}" + "".join(f"{k:<7}" for k in range(10)))
    F = []
    for s in seeds:
        c = per[s]
        f = [J(c[k], c[9]) for k in range(10)]
        F.append(f)
        print(f"{s:<6}" + "".join(f"{v:<7.2f}" for v in f))
    F = np.array(F)
    print(f"{'mean':<6}" + "".join(f"{v:<7.2f}" for v in F.mean(0)))


if __name__ == "__main__":
    main()
