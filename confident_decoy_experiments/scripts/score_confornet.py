#!/usr/bin/env python
"""Score ConforNet sweep poses: DockQ + interface pair_iptm, per (complex, interface, run, step, cn).

WHY A SEPARATE SCRIPT: decoy_set.py's pose() is bound to one pose layout
(PRED/<pdb>/seed_<s>/predictions/<pdb>_sample_<k>.cif) and to a fixed 15-interface set. ConforNet
poses are nested per run/step/confornet, so the enumeration differs entirely. The OST INVOCATION is
copied verbatim from decoy_set.py:pose() so the two are directly comparable.

--dockq ONLY, never --rigid-scores: OST's rigid-scores stage crashes on some assemblies
("3 points required for superposition") and takes a perfectly valid DockQ down with it. That bug
silently contaminated an earlier failure list (8c5h's real DockQ 0.899 was logged as 0).

CHAIN ROLES come from FoldBench/targets/interface_antibody_antigen.csv, per complex. They are NOT
uniform -- 8qf4 is antibody A / antigen B while 8aok, 8c5h, 8fxv, 8h63 are the reverse -- and
assuming a convention here would silently invert the antibody and antigen for half the set.

ONE OST CALL PER MODEL on the full native assembly, then the labelled pair is extracted BY NAME from
OST's own global chain mapping. Scoring each interface with its own call would let OST pick a
different predicted copy per interface on multi-copy assemblies.

Usage: .../python scripts/score_confornet.py [--jobs 24] [--complexes 8c5h,8qf4]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
ART = f"{EXP}/artifacts/confornet"
GT = "/home/ebrahim/FoldBench/examples/ground_truths"
OST = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/ebrahim/miniconda3/envs/foldbench/lib")
IFACE_CSV = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
CACHE = f"{EXP}/artifacts/confornet_ost"
THRESH = 0.23
os.makedirs(CACHE, exist_ok=True)


def _base(chain):
    """'A-2' -> 'A'. OST suffixes the Nth copy's chains in a multi-copy native assembly."""
    return chain.split("-")[0]


def interfaces():
    """{pdb: [(antibody_chain, antigen_chain), ...]} from the curated table. Raises if unreadable --
    a silently empty mapping would score nothing and look like a clean null result."""
    out = {}
    with open(IFACE_CSV) as fh:
        for row in csv.DictReader(fh):
            pdb = row["pdb_id"].split("-")[0]
            out.setdefault(pdb, []).append((row["antibody_chain"], row["antigen_chain"]))
    if not out:
        raise RuntimeError(f"no interfaces parsed from {IFACE_CSV}")
    return out


def poses_for(pdb, pfx="v3"):
    """Every ConforNet cif for one complex, with its (run, step, cn, sample) coordinates."""
    found = []
    for run_dir in sorted(os.listdir(ART)):
        if not run_dir.startswith(f"{pdb}_{pfx}_s"):
            continue
        run = int(run_dir.rsplit("_s", 1)[-1])
        dec = f"{ART}/{run_dir}/decoded"
        if not os.path.isdir(dec):
            continue
        for tag in sorted(os.listdir(dec)):
            if "_step" not in tag or "_cn" not in tag:
                continue
            step = int(tag.split("_step")[1].split("_")[0])
            cn = int(tag.split("_cn")[1])
            pdir = f"{dec}/{tag}/seed_{run}/predictions"
            if not os.path.isdir(pdir):
                continue
            for f in sorted(os.listdir(pdir)):
                if f.endswith(".cif") and "_sample_" in f:
                    smp = int(f.rsplit("_sample_", 1)[1].split(".")[0])
                    found.append((pdb, run, step, cn, smp, f"{pdir}/{f}"))
    return found


def score_one(arg):
    """One OST call -> a row per labelled interface. Cached by output path, so this is resumable."""
    pdb, run, step, cn, smp, cif = arg
    key = f"{pdb}_s{run}_step{step}_cn{cn}_sample{smp}"
    o = f"{CACHE}/{key}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        try:
            subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures",
                            "-m", cif, "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o,
                            "--fault-tolerant", "--min-pep-length", "4",
                            "--min-nuc-length", "4", "--dockq"],
                           env=ENV, capture_output=True, timeout=600)
        except Exception:
            return []
    try:
        d = json.load(open(o))
    except Exception:
        return []
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []

    # pair_iptm lives next to the cif, indexed by the MODEL chains OST actually mapped
    conf = cif.replace("_sample_", "_summary_confidence_sample_").replace(".cif", ".json")
    M = None
    try:
        M = json.load(open(conf))["chain_pair_iptm"]
    except Exception:
        pass

    rows = []
    for c1, c2 in IFACES.get(pdb, []):
        dq = pi = None
        for i, t in enumerate(ifs):
            # Strip the copy suffix before matching. In a multi-copy native assembly OST names the
            # second copy's chains "A-2"/"B-2", and it maps our single predicted copy to WHICHEVER
            # native copy fits best -- both are the same biological interface. Matching the labels
            # literally silently dropped 334 of 8fxv's 400 poses (and 45% of its free-run poses),
            # keeping only the ones OST happened to assign to copy 1. That is not a thinner sample,
            # it is a biased one, since the copy OST picks can correlate with pose quality.
            if {_base(t[0]), _base(t[1])} == {c1, c2} and i < len(qs):
                if dq is None or qs[i] > dq:      # if both copies match, keep the better
                    dq = qs[i]
                    if M:
                        a, b = ord(t[2]) - 65, ord(t[3]) - 65
                        pi = M[a][b] if (0 <= a < len(M) and 0 <= b < len(M)) else None
        if dq is None:      # pair absent from OST's mapping -> no score, never a guess
            continue
        rows.append([pdb, f"{c1}{c2}", run, step, cn, smp, round(dq, 4),
                     round(pi, 4) if pi is not None else ""])
    return rows


IFACES = interfaces()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24, help="OST calls in flight; all pinned to cores 0-31")
    ap.add_argument("--complexes", default="", help="comma-separated; default = every complete complex")
    ap.add_argument("--prefix", default="v3", help="tag generation to score, e.g. v3")
    ap.add_argument("--min-poses", type=int, default=400, help="only score complexes with >= this many")
    A = ap.parse_args()

    if A.complexes:
        pdbs = A.complexes.split(",")
    else:
        pdbs = sorted({d.split(f"_{A.prefix}_s")[0] for d in os.listdir(ART) if f"_{A.prefix}_s" in d})
        pdbs = [p for p in pdbs if p != "8che" and len(poses_for(p, A.prefix)) >= A.min_poses]

    work = []
    for p in pdbs:
        w = poses_for(p, A.prefix)
        print(f"{p}: {len(w)} poses, interfaces {IFACES.get(p, [])}", flush=True)
        work += w
    print(f"\nscoring {len(work)} poses across {len(pdbs)} complexes with {A.jobs} jobs", flush=True)

    out = f"{EXP}/artifacts/confornet_scores_{A.prefix}.csv"
    n = 0
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["pdb", "iface", "run", "step", "cn", "sample", "dockq", "pair_iptm"])
        with Pool(A.jobs) as pool:
            for rows in pool.imap_unordered(score_one, work, chunksize=4):
                for r in rows:
                    wr.writerow(r)
                    n += 1
                if n and n % 500 == 0:
                    fh.flush()
                    print(f"  {n} rows", flush=True)
    print(f"\nwrote {n} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
