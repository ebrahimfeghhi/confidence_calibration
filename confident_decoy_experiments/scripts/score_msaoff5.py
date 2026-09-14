#!/usr/bin/env python
"""Score the MSA-off-then-on arm: DockQ + interface pair_iptm, per (complex, seed, sample).

The arm: PROTENIX_MSA_FROM_CYCLE=5 -- the MSA module is skipped for recycles 0-4 and active for 5-9,
so the trunk forms z from templates + sequence alone before any coevolutionary signal arrives.
Everything else is stock (MSA subsampling ON, MC dropout ON), and the seeds are 101-150, the SAME
range as the free run, so the two arms are matched pose-for-pose and seed-for-seed.

Scoring protocol is copied from score_confornet.py so every arm in this project is comparable:
--dockq only (never --rigid-scores, which crashes on some assemblies and takes a valid DockQ with it),
chain roles from the curated FoldBench table, one OST call per model against the full native assembly,
labelled pair extracted BY NAME with the multi-copy "-2" suffix stripped.

Layout: artifacts/sweep_out/msaoff5/msaoff5_<pdb>/<pdb>/seed_<s>/predictions/<pdb>_sample_<k>.cif

Usage: .../python scripts/score_msaoff5.py [--jobs 24] [--min-poses 250]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
# ARM is set from --arm so this one scorer serves every sweep with the same layout
# (msaoff5, phase30, ...) instead of forking a near-identical script per arm.
ARM = "msaoff5"
ART = f"{EXP}/artifacts/sweep_out/{ARM}"
GT = "/home/ebrahim/FoldBench/examples/ground_truths"
OST = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/ebrahim/miniconda3/envs/foldbench/lib")
IFACE_CSV = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
CACHE = f"{EXP}/artifacts/{ARM}_ost"
os.makedirs(CACHE, exist_ok=True)


def _base(c):
    """'A-2' -> 'A'. OST suffixes the Nth copy's chains in a multi-copy native assembly; matching the
    labels literally silently drops every pose it mapped to copy 2 (cost 8fxv 334/400 once)."""
    return c.split("-")[0]


def interfaces():
    out = {}
    with open(IFACE_CSV) as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["pdb_id"].split("-")[0], []).append(
                (row["antibody_chain"], row["antigen_chain"]))
    if not out:
        raise RuntimeError(f"no interfaces parsed from {IFACE_CSV}")
    return out


IFACES = interfaces()


def poses_for(pdb):
    root = f"{ART}/{ARM}_{pdb}/{pdb}"
    found = []
    if not os.path.isdir(root):
        return found
    for sd in sorted(os.listdir(root)):
        m = re.match(r"seed_(\d+)$", sd)
        if not m:
            continue
        seed = int(m.group(1))
        pdir = f"{root}/{sd}/predictions"
        if not os.path.isdir(pdir):
            continue
        for f in sorted(os.listdir(pdir)):
            if f.endswith(".cif") and "_sample_" in f:
                found.append((pdb, seed, int(f.rsplit("_sample_", 1)[1].split(".")[0]),
                              f"{pdir}/{f}"))
    return found


def score_one(arg):
    pdb, seed, smp, cif = arg
    o = f"{CACHE}/{pdb}_{seed}_{smp}.json"
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
            if {_base(t[0]), _base(t[1])} == {c1, c2} and i < len(qs):
                if dq is None or qs[i] > dq:
                    dq = qs[i]
                    if M:
                        a, b = ord(t[2]) - 65, ord(t[3]) - 65
                        pi = M[a][b] if (0 <= a < len(M) and 0 <= b < len(M)) else None
        if dq is None:
            continue
        rows.append([pdb, f"{c1}{c2}", seed, smp, round(dq, 4),
                     round(pi, 4) if pi is not None else ""])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--arm", default="msaoff5",
                    help="sweep_out subdirectory / output prefix, e.g. msaoff5 or phase30")
    ap.add_argument("--complexes", default="")
    ap.add_argument("--min-poses", type=int, default=250,
                    help="only score complexes that finished all 50 seeds; a partial complex would "
                         "enter the averages as a shallower arm")
    A = ap.parse_args()

    global ARM, ART, CACHE
    ARM = A.arm
    ART = f"{EXP}/artifacts/sweep_out/{ARM}"
    CACHE = f"{EXP}/artifacts/{ARM}_ost"
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.isdir(ART):
        raise SystemExit(f"no such arm directory: {ART}")

    if A.complexes:
        pdbs = A.complexes.split(",")
    else:
        pdbs = sorted(d.replace(f"{ARM}_", "") for d in os.listdir(ART) if d.startswith(f"{ARM}_"))
        pdbs = [p for p in pdbs if len(poses_for(p)) >= A.min_poses]

    work = []
    for p in pdbs:
        w = poses_for(p)
        print(f"{p}: {len(w)} poses, interfaces {IFACES.get(p, [])}", flush=True)
        work += w
    print(f"\nscoring {len(work)} poses across {len(pdbs)} complexes, {A.jobs} jobs", flush=True)

    out = f"{EXP}/artifacts/{ARM}_scores.csv"
    n = 0
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["pdb", "iface", "seed", "sample", "dockq", "pair_iptm"])
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
