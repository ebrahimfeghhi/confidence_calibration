#!/usr/bin/env python
"""Score generational self-templating poses: DockQ + interface pair_iptm, per (complex, gen, lineage).

Same OST invocation and the same multi-copy chain fix as score_confornet.py, so the arms are directly
comparable: --dockq only (never --rigid-scores, which crashes on some assemblies and takes a valid
DockQ down with it), chain roles from the curated FoldBench table, one OST call per model on the full
native assembly with the labelled pair extracted by name.

GENERATION 1 IS THE FREE RUN. V1 was not regenerated -- it reuses the existing seeds 101-110 from
predictions_template -- so its poses live in a different directory and are scored from there. That
makes gen 1 exactly the free-run baseline and gens 2-5 the self-templated arm, on the same seeds.

Usage: .../python scripts/score_selftmpl.py [--jobs 24] [--min-poses 200]
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
ART = f"{EXP}/artifacts/selftmpl"
FREE = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
GT = "/home/ebrahim/FoldBench/examples/ground_truths"
OST = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/ebrahim/miniconda3/envs/foldbench/lib")
IFACE_CSV = "/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv"
CACHE = f"{EXP}/artifacts/selftmpl_ost"
SEED0, LINEAGES = 101, 10
os.makedirs(CACHE, exist_ok=True)


def _base(c):
    """'A-2' -> 'A'. OST suffixes the Nth copy's chains in a multi-copy native assembly, and matching
    the labels literally silently drops every pose it mapped to copy 2."""
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
    """(pdb, gen, lineage, sample, cif). gen 1 = the reused free run; gens 2-5 = self-templated."""
    found = []
    for lin in range(LINEAGES):                       # gen 1: free run, seeds 101-110
        s = SEED0 + lin
        d = f"{FREE}/{pdb}/seed_{s}/predictions"
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".cif") and "_sample_" in f:
                    found.append((pdb, 1, lin, int(f.rsplit("_sample_", 1)[1].split(".")[0]),
                                  f"{d}/{f}"))
    dec = f"{ART}/{pdb}_gen/decoded"                   # gens 2-5
    if os.path.isdir(dec):
        for tag in sorted(os.listdir(dec)):
            m = re.search(r"_g(\d+)_l(\d+)$", tag)
            if not m:
                continue
            g, lin = int(m.group(1)), int(m.group(2))
            pdir = f"{dec}/{tag}/seed_{SEED0 + g * LINEAGES + lin}/predictions"
            if not os.path.isdir(pdir):
                continue
            for f in sorted(os.listdir(pdir)):
                if f.endswith(".cif") and "_sample_" in f:
                    found.append((pdb, g + 1, lin,          # g is 0-indexed in the filename; report V-number
                                  int(f.rsplit("_sample_", 1)[1].split(".")[0]), f"{pdir}/{f}"))
    return found


def score_one(arg):
    pdb, gen, lin, smp, cif = arg
    o = f"{CACHE}/{pdb}_g{gen}_l{lin}_s{smp}.json"
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
        rows.append([pdb, f"{c1}{c2}", gen, lin, smp, round(dq, 4),
                     round(pi, 4) if pi is not None else ""])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--complexes", default="")
    ap.add_argument("--min-poses", type=int, default=200,
                    help="gens 2-5 must be complete (200) before a complex is scored; a partial "
                         "complex would enter the averages as a shallower arm")
    A = ap.parse_args()

    if A.complexes:
        pdbs = A.complexes.split(",")
    else:
        pdbs = []
        for d in sorted(os.listdir(ART)):
            if not d.endswith("_gen"):
                continue
            p = d[:-4]
            n = len([1 for x in poses_for(p) if x[1] > 1])
            if n >= A.min_poses:
                pdbs.append(p)

    work = []
    for p in pdbs:
        w = poses_for(p)
        print(f"{p}: {len(w)} poses ({len([x for x in w if x[1]==1])} free V1 + "
              f"{len([x for x in w if x[1]>1])} self-templated), interfaces {IFACES.get(p, [])}",
              flush=True)
        work += w
    print(f"\nscoring {len(work)} poses across {len(pdbs)} complexes, {A.jobs} jobs", flush=True)

    out = f"{EXP}/artifacts/selftmpl_scores.csv"
    n = 0
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["pdb", "iface", "gen", "lineage", "sample", "dockq", "pair_iptm"])
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
