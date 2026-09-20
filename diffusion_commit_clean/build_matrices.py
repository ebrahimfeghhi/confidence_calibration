#!/usr/bin/env python
"""Compute the three pose-similarity matrices for every interface, so cluster purity can decide which
metric defines a "basin".

ONE FILE PER INTERFACE, an .npz holding everything needed to evaluate it later:

    epitope, fcc, dockq   (N, N) float32 similarity, symmetric, diagonal 1
    seed, sample          (N,)   which pose each row is
    model_ab, model_ag    (N,)   the chains actually compared, from chain_map.py's mapping
    dockq_gt, pair_iptm   (N,)   ground truth and confidence, the labels purity is scored against

Keeping the labels beside the matrices is deliberate: a matrix whose row order has drifted from its
labels is a silent, unfalsifiable bug, and row order here is nothing more than a list index.

RECOMPUTED FROM SCRATCH, not reused. 55 cached matrices exist from before the chain mapping, and 46%
of all poses turn out to have a permuted mapping, so some unknown subset of those was built on the
wrong chains. Identifying which would cost as much as redoing them.

POSES WITH NO CONTACTS ARE DROPPED. On a multi-copy assembly a pose can bind a different antigen copy
than the interface asks about, leaving zero antibody-antigen contacts for the pair we resolved (1 of
12 on 8w5d_AB). Its similarity to everything is 0 by construction, which is not a basin -- it is an
absence of an interface. Dropping is done BEFORE any matrix is built so all three share a row order,
and the count is recorded per interface rather than left to be inferred from a row count.

    build_matrices.py --out DIR [--ifaces 8w5d_AB,...] [--device cuda]

Resumable: an interface whose .npz exists is skipped.
"""
import argparse
import collections
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402
import paths  # noqa: E402


def load_pool():
    """{interface: [(seed, sample, dockq_gt, pair_iptm), ...]}"""
    pool = collections.defaultdict(list)
    with open(str(paths.PXM2025_POOL_CSV)) as fh:
        for r in csv.DictReader(fh):
            pool[f"{r['pdb']}_{r['chain1']}{r['chain2']}"].append(
                (r["seed"], int(r["sample"]), float(r["dockq"]), float(r["pair_iptm"])))
    return pool


def build(iface, rows, maps, roles, device):
    """Three matrices plus aligned labels for one interface, or None if too few usable poses."""
    items, nomap = metrics.items_for_interface(iface, [(s, x) for s, x, _, _ in rows], maps, roles)
    # items_for_interface drops unmapped poses, so re-pair labels to what survived rather than
    # assuming the two lists still line up
    by_key = {(s, x): (d, i) for s, x, d, i in rows}
    keys = [(s, x) for (s, x) in [(s, x) for s, x, _, _ in rows]
            if (s, x) not in set(nomap)]
    # Filter at the CONTACT cutoff (5.0 A), not the epitope cutoff (4.5 A). Using 4.5 dropped ~1,110
    # poses that DockQ scores as real interfaces: on ring assemblies like 9kmp (5 antigens, 5
    # antibodies, each antigen bound by two) the second interface of the ring is a glancing 4.70 A
    # contact -- outside 4.5, inside DockQ's 5.0 -- and 162 of those poses had GT DockQ above 0.80.
    keep = [(it, k) for it, k in zip(items, keys)
            if len(metrics.epitope_set(it[0], it[1], it[2], cutoff=metrics.CONTACT_CUTOFF)) > 0]
    nocontact = len(items) - len(keep)
    if len(keep) < 5:
        return None, len(nomap), nocontact
    items = [it for it, _ in keep]
    keys = [k for _, k in keep]
    M, _ = metrics.set_matrix(items, "epitope")
    F, _ = metrics.set_matrix(items, "fcc")
    D = metrics.dockq_pose_matrix(items, device=device)
    out = dict(
        epitope=M, fcc=F, dockq=D,
        seed=np.array([k[0] for k in keys]),
        sample=np.array([k[1] for k in keys], dtype=np.int16),
        model_ab=np.array([it[1] for it in items]),
        model_ag=np.array([it[2] for it in items]),
        dockq_gt=np.array([by_key[k][0] for k in keys], dtype=np.float32),
        pair_iptm=np.array([by_key[k][1] for k in keys], dtype=np.float32),
    )
    return out, len(nomap), nocontact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ifaces", default=None, help="comma-separated subset")
    ap.add_argument("--device", default="cuda")
    A = ap.parse_args()

    os.makedirs(A.out, exist_ok=True)
    pool, maps, roles = load_pool(), metrics.load_chain_maps(), metrics.load_roles()
    want = set(A.ifaces.split(",")) if A.ifaces else None
    todo = sorted(k for k in pool if (not want or k in want) and k in roles)
    todo = [k for k in todo if not os.path.exists(f"{A.out}/{k}.npz")]
    print(f"  interfaces to build: {len(todo)}", flush=True)

    log = open(f"{A.out}/build_log.csv", "a", newline="")
    w = csv.writer(log)
    if log.tell() == 0:
        w.writerow(["interface", "poses_in_pool", "rows", "dropped_unmapped",
                    "dropped_no_contact", "seconds"])
    t0 = time.time()
    for n, iface in enumerate(todo, 1):
        t = time.time()
        try:
            out, nomap, nocon = build(iface, pool[iface], maps, roles, A.device)
        except Exception as e:
            print(f"  {iface}: FAILED {type(e).__name__}: {e}", flush=True)
            w.writerow([iface, len(pool[iface]), 0, "", "", round(time.time() - t, 1)])
            log.flush()
            continue
        nrow = 0 if out is None else len(out["seed"])
        if out is not None:
            np.savez_compressed(f"{A.out}/{iface}.npz", **out)
        w.writerow([iface, len(pool[iface]), nrow, nomap, nocon, round(time.time() - t, 1)])
        log.flush()
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el/n:.1f}s/iface  eta {(len(todo)-n)*el/n/60:.0f}m", flush=True)
    log.close()
    print(f"  done in {(time.time()-t0)/60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
