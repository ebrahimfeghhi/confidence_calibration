#!/usr/bin/env python
"""For every 9x1h pose, find which (light, heavy, antigen) chains form a mutually-docked protomer.

WHY THIS IS NOT OPTIONAL. 9x1h is a 2-copy assembly: chains A,B are the light chain entity, C,D the
heavy, E,F the antigen. build_multitmpl.py matches donor chains to query entities by SEQUENCE, and
every copy has the identical sequence, so the pick among ties is arbitrary. On the two donors examined
by hand the real protomers were (A,C,F) and (B,D,E) -- A-C 209 contacts, C-F 55, A-F 53, but A-E only
17. The arbitrary pick of A,C,E would pair a Fab with the antigen copy it is NOT bound to.

With the cross-chain mask ON that is harmless (only intra-chain folds are read). With the mask OFF --
which is the whole point of the rescoring run -- it hands the model a nonsense arrangement. And the
topology CANNOT be assumed constant across poses: which copy a Fab docks to is exactly the kind of
thing that varies between samples, so it is measured per pose here.

SELECTION RULE: score every (light, heavy, antigen) triple by the total heavy-atom contacts across its
three pairs (5 A cutoff) and keep the best. A pose whose best triple is weakly connected is REPORTED,
not silently used -- a Fab that touches no antigen has no dock to transmit.

Usage: .../python scripts/pose_protomers.py [--jobs 24] [--out artifacts/9x1h_protomers.csv]
Outputs: seed,sample,light,heavy,antigen,lh_contacts,la_contacts,ha_contacts,total,ok
"""
import argparse
import csv
import glob
import itertools
import os
import re
from multiprocessing import Pool

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
# entity -> the two model chains carrying it, by residue count (light 220, heavy 225, antigen 290)
GROUPS = {"light": ("A", "B"), "heavy": ("C", "D"), "antigen": ("E", "F")}
CUTOFF = 5.0


def one(cif):
    import gemmi
    m = re.search(r"seed_(\d+)/predictions/\w+_sample_(\d+)\.cif$", cif)
    if not m:
        return None
    seed, smp = int(m.group(1)), int(m.group(2))
    st = gemmi.read_structure(cif)
    st.setup_entities()
    st.remove_ligands_and_waters()
    pos = {}
    for c in st[0]:
        if len(c) < 20:
            continue
        pos[c.name] = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in c for a in r
                                if a.element.name != "H"])
    def contacts(a, b):
        if a not in pos or b not in pos:
            return 0
        d = np.linalg.norm(pos[a][:, None, :] - pos[b][None, :, :], axis=-1)
        return int((d.min(axis=1) < CUTOFF).sum())
    best = None
    for L, H, Ag in itertools.product(*GROUPS.values()):
        lh, la, ha = contacts(L, H), contacts(L, Ag), contacts(H, Ag)
        tot = lh + la + ha
        # NOTE best[-1] is `tot`; an earlier version compared best[-2] which is `ha`,
        # so it maximised heavy-antigen contacts alone and returned the same triple for
        # all 250 poses, contradicting the two verified by hand.
        if best is None or tot > best[-1]:
            best = (L, H, Ag, lh, la, ha, tot)
    L, H, Ag, lh, la, ha, tot = best
    # a usable donor needs the Fab assembled AND touching the antigen
    ok = int(lh >= 50 and (la + ha) >= 20)
    return [seed, smp, L, H, Ag, lh, la, ha, tot, ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--out", default=f"{EXP}/artifacts/9x1h_protomers.csv")
    A = ap.parse_args()
    cifs = sorted(glob.glob(f"{P}/out/9x1h/9x1h/seed_*/predictions/9x1h_sample_*.cif"))
    print(f"{len(cifs)} poses", flush=True)
    with Pool(A.jobs) as pool:
        rows = [r for r in pool.map(one, cifs) if r]
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "sample", "light", "heavy", "antigen",
                    "lh_contacts", "la_contacts", "ha_contacts", "total", "ok"])
        w.writerows(sorted(rows))
    import collections
    combos = collections.Counter(tuple(r[2:5]) for r in rows)
    print(f"\nwrote {A.out}: {len(rows)} poses, {sum(r[-1] for r in rows)} usable")
    print("protomer combinations found:")
    for k, v in combos.most_common():
        print(f"   {'/'.join(k)}: {v}")
    bad = [r for r in rows if not r[-1]]
    if bad:
        print(f"\n{len(bad)} poses REJECTED (Fab not assembled or not touching antigen):")
        for r in bad[:8]:
            print(f"   seed {r[0]} sample {r[1]}: {r[2]}/{r[3]}/{r[4]} "
                  f"lh={r[5]} la={r[6]} ha={r[7]}")


if __name__ == "__main__":
    main()
