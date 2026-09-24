#!/usr/bin/env python
"""STEP 1 of the method: the untemplated pose pool that everything else selects from.

The method never generates poses of its own until step 11 -- it CHOOSES which of these to feed back in
as templates. So this is the input, and its two numbers set the ceiling on everything downstream:

    seeds    how many independent folds per complex. ipTM is largely a property of the SEED (a seed
             commits to a docking basin early and stays there), so seeds are what buys pose diversity.
             Samples within a seed are far more alike than poses across seeds.
    samples  poses per fold, free apart from the diffusion sampling itself.

The published pool was 50 seeds x 5 samples = 250 poses per complex, which is what the results assume.
Changing either invalidates comparison against them, so both are explicit arguments rather than
constants buried in a driver.

A free unit needs nothing but the complex's Protenix input json -- no templates, no namespace, no
reference. fold.py reads `<pdb>-final-updated.json` from the dataset's jobs dir and runs it as-is,
which is why this script only has to emit a queue.

    build_free_queue.py --out queue_free.tsv [--pdbs 8ror,9k94] [--seeds 50] [--start-seed 300000]

then hand it to the driver, which is the same one every other step uses:

    fold.py --queue queue_free.tsv --out <dir> --csv <dir>/results.csv --claim-dir <dir>/claims
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402


def complexes(dataset, want=None):
    """Every complex the dataset has an input json for, optionally restricted to `want`."""
    jobs, _, _ = paths.dataset(dataset)
    out = []
    for p in sorted(glob.glob(f"{jobs}/*-final-updated.json")):
        pdb = os.path.basename(p).split("-")[0]
        if not want or pdb in want:
            out.append(pdb)
    return out


def free_units(pdbs, seeds, start_seed):
    """One queue line per (complex, seed).

    Columns are the driver's: kind, pdb, target, tag, seed. `target` is "*" because a free fold has no
    interface it was chosen FOR -- it is scored against every interface of the complex. The tag is
    constant: fold.py identifies a unit as (seed, tag), so one tag with N seeds is already N distinct
    units, and a per-seed tag would only duplicate information the seed column already carries.
    """
    for pdb in pdbs:
        for i in range(seeds):
            yield "\t".join(["free", pdb, "*", "free", str(start_seed + i)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdbs", default=None, help="comma-separated subset; default is every complex")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=300000)
    ap.add_argument("--dataset", choices=sorted(paths.DATASET_NAMES), default="pxm2025")
    A = ap.parse_args()

    pdbs = complexes(A.dataset, set(A.pdbs.split(",")) if A.pdbs else None)
    if not pdbs:
        raise SystemExit("no complexes matched --pdbs")
    lines = list(free_units(pdbs, A.seeds, A.start_seed))
    os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
    open(A.out, "w").write("\n".join(lines) + "\n")
    print(f"  complexes: {len(pdbs)}   seeds each: {A.seeds}   folds: {len(lines)}")
    print(f"  poses produced: {len(lines)} x nsample (default 5) = {len(lines) * 5}")
    print(f"  -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
