#!/usr/bin/env python
"""Pick the experiment: 20 confident-decoy interfaces, and for each, 5 good seeds and 5 confident
wrong seeds.

THE QUESTION. On a confident decoy the pool contains a usable pose and ipTM ships a bad one, so the
failure is selection. Self-templating each seed on its own poses should, if the mechanism is real,
raise a GOOD seed's confidence more than a WRONG seed's -- making the good one win. Seen once
(+0.194 against +0.014 on 9h01_AB); this picks the sample to test whether it generalises.

WHY 5 AND 5 PER INTERFACE, paired. Comparing templated against free across interfaces confounds the
effect with how hard each target is. Within one interface, the good and wrong seeds face the same
target, the same sequence and the same MSA, so the only difference is which basin the seed fell into.

DEFINITIONS
  good seed    contains at least one pose with DockQ >= 0.23 -- the answer is in there somewhere
  wrong seed   its own top-ipTM pose is below 0.23 AND its ipTM is high, so it is what currently
               outranks the good seeds. Not merely "a bad seed": the competition.

ONE INTERFACE PER COMPLEX. 9vor contributes four interfaces and 9g9m/9g9n/9g9o/9g9p are near
duplicates; treating those as independent observations would inflate n without adding evidence.

    select_seeds.py --out plan.tsv [--n 20] [--max-tokens 900]
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402

ACC = 0.23


def load_pool():
    """{interface: {seed: [(iptm, dockq)]}} from the scored free pool."""
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(str(paths.PXM2025_POOL_CSV))):
        iface = f"{r['pdb']}_{r['chain1']}{r['chain2']}"
        pool[iface][r["seed"]].append((float(r["pair_iptm"]), float(r["dockq"])))
    return pool


def tokens():
    """Token count per pdb, for excluding complexes whose folds take tens of minutes."""
    t = {}
    p = f"{paths.DATA_DIR}/archive/hybrid25/units.csv"
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            t[r["pdb"]] = int(r["tokens"])
    return t


def system_key(pdb):
    """A hash of the complex's SEQUENCES, so re-depositions of one system collapse together.

    9g9m/9g9n/9g9o/9g9p have byte-identical sequences for all three chains -- the same antigen and
    antibody deposited four times. Deduplicating by pdb code kept all four, which would have folded
    one problem four times and counted it as four independent interfaces.
    """
    d = json.load(open(f"{paths.PXM2025_JOBS}/{pdb}-final-updated.json"))
    d = d if isinstance(d, list) else [d]
    seqs = sorted(v["sequence"] for s in d[0]["sequences"] for v in s.values()
                  if isinstance(v, dict) and "sequence" in v)
    return hashlib.md5("|".join(seqs).encode()).hexdigest()[:12]


def is_confident_decoy(seeds):
    allp = [p for v in seeds.values() for p in v]
    if not allp:
        return False
    # the single most confident pose in the whole pool is bad, yet a usable pose exists
    return max(allp)[1] < ACC and any(d >= ACC for _, d in allp)


def donor_dir(pdb, seed):
    """Where this seed's free poses live. The pool folds carry the legacy marker."""
    return f"{paths.FREE_DIR}/{pdb}/{seed}-legacy-no-provenance/predictions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=900)
    A = ap.parse_args()

    pool, tok = load_pool(), tokens()
    cands = []
    for iface, seeds in pool.items():
        pdb = iface.split("_")[0]
        if len(seeds) < 50 or not is_confident_decoy(seeds):
            continue
        if tok.get(pdb, 10 ** 9) > A.max_tokens:
            continue
        good = sorted((s for s, v in seeds.items() if max(d for _, d in v) >= ACC),
                      key=lambda s: -max(d for _, d in seeds[s]))
        wrong = sorted((s for s, v in seeds.items()
                        if max(seeds[s], key=lambda x: x[0])[1] < ACC),
                       key=lambda s: -max(i for i, _ in seeds[s]))
        # poses must actually be on disk to template on
        good = [s for s in good if os.path.isdir(donor_dir(pdb, s))]
        wrong = [s for s in wrong if os.path.isdir(donor_dir(pdb, s))]
        if len(good) >= A.per_class and len(wrong) >= A.per_class:
            cands.append((pdb, iface, tok.get(pdb, 0), good[:A.per_class], wrong[:A.per_class]))

    # one interface per SYSTEM (by sequence, not by pdb code), smallest first for fast folds
    cands.sort(key=lambda c: c[2])
    seen, picked = set(), []
    for pdb, iface, t, good, wrong in cands:
        key = system_key(pdb)
        if key in seen:
            continue
        seen.add(key)
        picked.append((pdb, iface, t, good, wrong))
        if len(picked) == A.n:
            break

    with open(A.out, "w") as fh:
        fh.write("pdb\tinterface\ttokens\tclass\tseed\n")
        for pdb, iface, t, good, wrong in picked:
            for s in good:
                fh.write(f"{pdb}\t{iface}\t{t}\tgood\t{s}\n")
            for s in wrong:
                fh.write(f"{pdb}\t{iface}\t{t}\twrong\t{s}\n")
    n_folds = sum(len(g) + len(w) for _, _, _, g, w in picked)
    print(f"  interfaces (distinct complexes): {len(picked)}")
    print(f"  folds to run: {n_folds}  ({A.per_class} good + {A.per_class} wrong each)")
    print(f"  tokens {min(c[2] for c in picked)}-{max(c[2] for c in picked)}")
    print(f"  wrote {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
