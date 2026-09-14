#!/usr/bin/env python
"""Score the protein-protein sweep with OST, the same way the antibody-antigen set was scored.

WHY IT EXISTS. The whole point of the protein-protein run is the control for the seed-scaling question:
scaling buys ~0.2 DockQ on antibody-antigen and <0.04 on protein-protein, and we want to know whether
the epitope-search mechanism we found on ab-ag is absent here. That comparison is only meaningful if
both sides are scored by the identical pipeline.

GROUND TRUTH. artifacts/pp/gt/<pdb>-assembly1.cif, downloaded from RCSB by ops/fetch_pp_gt.sh.
Verified byte-identical to the files FoldBench references in examples/ground_truths on 8gy5, 8qf4,
8c5h and 7st5, so this is FoldBench's ground truth and not a substitute for it.

OST FLAGS. `--dockq` only. Adding --rigid-scores makes OST abort with "3 points required for
superposition" on some targets and throw away an otherwise valid DockQ -- that bug silently turned
real scores into zeros once already (8c5h logged 0.0 when it really scores 0.899).

CHAIN NAMES. interface_chain_id_1/2 in FoldBench's targets csv are REFERENCE (native) chain names, as
on the antibody-antigen side -- see scripts/iface_chains.py. Interfaces are matched against OST's
`dockq_interfaces` rows, whose layout is [ref1, ref2, mdl1, mdl2], and the MODEL chains are read from
that same row so the confidence lookup cannot drift from the DockQ.

Results are flushed continuously, so progress is inspectable while it runs.

Usage:
  /home/ebrahim/miniconda3/envs/foldbench/bin/python scripts/score_pp.py [--jobs 32] [--limit N]
Outputs artifacts/pp/dockq_pp.csv: pdb,chain1,chain2,seed,sample,dockq,fnat,irmsd,lrmsd,pair_iptm
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
PP = f"{EXP}/artifacts/pp"
GT = f"{PP}/gt"
OSTD = f"{PP}/ost"
IFACE_CSV = "/home/ebrahim/FoldBench/targets/interface_protein_protein.csv"
OST = "/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/ebrahim/miniconda3/envs/foldbench/lib")
HDR = ["pdb", "chain1", "chain2", "seed", "sample", "dockq", "fnat", "irmsd", "lrmsd", "pair_iptm"]


def run_ost(model, native, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    try:
        subprocess.run([OST, "compare-structures", "-m", model, "-r", native, "-o", out,
                        "--fault-tolerant", "--min-pep-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        pass


def chain_idx(ch):
    """Model chain label -> index into chain_pair_iptm ('A'->0). Leading alphabetic run only, so a
    digit-suffixed chain like 'A0' maps to 0 rather than ord('0')-based garbage."""
    base, alpha = ch.split("-")[0], ""
    for c in base:
        if not c.isalpha():
            break
        alpha += c
    idx = 0
    for c in alpha:
        idx = idx * 26 + (ord(c.upper()) - ord("A") + 1)
    return idx - 1


def _base(ch):
    """Strip OST's copy suffix ('A-2' -> 'A'). Needed because OST may report an interface on the
    OTHER symmetry copy than the targets csv names -- 8a51 asks ref A|B and OST reports A-2|B-2, the
    same interface in the other copy. Compared as a SORTED MULTISET so a genuine symmetry interface
    (A|A-2 -> ['A','A']) still cannot match a hetero one (A|B -> ['A','B'])."""
    return re.sub(r"-\d+$", "", ch)


def extract(ost_json, conf_json, c1, c2):
    """-> (dockq, fnat, irmsd, lrmsd, pair_iptm) for the interface {c1, c2}, or None."""
    try:
        d = json.load(open(ost_json))
    except Exception:
        return None
    want = sorted([_base(c1), _base(c2)])
    for k, row in enumerate(d.get("dockq_interfaces") or []):
        if len(row) < 4:
            continue
        if {row[0], row[1]} != {c1, c2} and sorted([_base(row[0]), _base(row[1])]) != want:
            continue

        def at(key):
            v = d.get(key)
            return v[k] if isinstance(v, list) and k < len(v) else None

        pi = None
        try:
            M = json.load(open(conf_json)).get("chain_pair_iptm")
            i, j = chain_idx(row[2]), chain_idx(row[3])
            if M and 0 <= i < len(M) and 0 <= j < len(M):
                pi = M[i][j]
        except Exception:
            pass
        return at("dockq"), at("fnat"), at("irmsd"), at("lrmsd"), pi
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only the first N complexes (smoke test)")
    ap.add_argument("--out", default=f"{PP}/dockq_pp.csv")
    A = ap.parse_args()
    os.makedirs(OSTD, exist_ok=True)

    ifm = collections.defaultdict(list)
    for r in csv.DictReader(open(IFACE_CSV)):
        ifm[r["pdb_id"].split("-")[0]].append((r["interface_chain_id_1"],
                                               r["interface_chain_id_2"]))

    # only complexes that actually finished (250/250 poses); a partial one would bias any mean
    todo, partial, nogt = [], 0, []
    for d in sorted(glob.glob(f"{PP}/out/*/")):
        pdb = os.path.basename(d.rstrip("/"))
        # predictions nest as out/<pdb>/<pdb>/seed_*/predictions/
        poses = glob.glob(f"{d}/**/seed_*/predictions/*_sample_*.cif", recursive=True)
        if len(poses) < 250:
            partial += 1
            continue
        if not os.path.exists(f"{GT}/{pdb}-assembly1.cif"):
            nogt.append(pdb)
            continue
        todo.append((pdb, poses))
    if A.limit:
        todo = todo[:A.limit]
    print(f"complete complexes: {len(todo)}  (still running: {partial}, no ground truth: {nogt})",
          flush=True)

    jobs = []
    for pdb, poses in todo:
        nat = f"{GT}/{pdb}-assembly1.cif"
        for cif in sorted(poses):
            m = re.search(r"seed_(\d+)/predictions/.*_sample_(\d+)\.cif$", cif)
            if not m:
                continue
            seed, smp = m.group(1), m.group(2)
            conf = cif.replace(f"_sample_{smp}.cif", f"_summary_confidence_sample_{smp}.json")
            jobs.append((pdb, cif, conf, nat, f"{OSTD}/{pdb}_{seed}_{smp}.json", seed, smp))
    print(f"OST on {len(jobs)} poses, {A.jobs} workers", flush=True)

    done = [0]

    def work(j):
        run_ost(j[1], j[3], j[4])
        done[0] += 1
        if done[0] % 2000 == 0:
            print(f"  ost {done[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=A.jobs) as ex:
        list(ex.map(work, jobs))

    prev = {}
    if os.path.exists(A.out):
        for r in csv.DictReader(open(A.out)):
            prev[(r["pdb"], r["chain1"], r["chain2"], r["seed"], r["sample"])] = [r[h] for h in HDR]
    n, miss = 0, collections.Counter()
    for pdb, cif, conf, nat, oj, seed, smp in jobs:
        for c1, c2 in ifm.get(pdb, []):
            got = extract(oj, conf, c1, c2)
            if got is None or got[0] is None:
                miss[pdb] += 1
                continue
            dq, fn, ir, lr, pi = got
            prev[(pdb, c1, c2, seed, smp)] = [pdb, c1, c2, seed, smp, dq, fn, ir, lr, pi]
            n += 1
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(prev[k] for k in sorted(prev))
    print(f"\nscored {n} pose-interfaces, wrote {len(prev)} rows -> {A.out}", flush=True)
    if miss:
        print("poses with no matching interface in OST output (top 10):", flush=True)
        for k, v in miss.most_common(10):
            print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
