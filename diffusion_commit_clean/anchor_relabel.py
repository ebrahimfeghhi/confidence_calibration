#!/usr/bin/env python
"""Relabel every pose against a POSE instead of against the crystal.

WHY. The running selection sorts poses separately per interface, so it must know which model chain
pair is "interface A-B" in every pose. chain_map.py answers that by aligning each pose to the CRYSTAL,
which means the selection as implemented cannot be run at test time. This script answers the same
question using one of the poses as the reference, so nothing about the native structure is needed.

THE ANCHOR IS THE POSE WITH THE HIGHEST MEAN pair_iptm over the complex's interfaces. One anchor per
COMPLEX, not per interface. That is not a cost saving, it is a correctness property: a single
anchor->pose mapping is a bijection on chains, so the anchor's interfaces (A,B) and (A,C) are
guaranteed to land on DIFFERENT chain pairs of the pose. With a separate anchor per interface the two
mappings are computed independently and nothing stops both from selecting the pose's (A,B) pair -- one
physical interface would enter two keep lists as if it were two distinct hypotheses, silently.

WHERE THE CRYSTAL STILL APPEARS, AND WHY IT IS NOT CHEATING. The selection never uses it. It is used
once, for REPORTING only: to decide which of the anchor's chain pairs should be called "8w5d_AB" so
that the donor set can be lined up against the crystal-labelled run and diffed. At test time the slots
would simply be unnamed, and the algorithm is indifferent to their names. Deleting this step changes
no donor, only the column header it is filed under.

OUTPUT is drop-in for chain_map.py's: a chain_maps csv whose `chain_mapping` is {anchor_chain:
pose_chain} instead of {ref_chain: pose_chain}, plus a chain_roles csv naming the ANCHOR's chains for
each interface. Feed both to build_matrices.py --maps/--roles.

    anchor_relabel.py --out-dir DIR [--pdbs 8ror,9puo] [--jobs 8]
"""
import argparse
import collections
import csv
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402

logging.disable(logging.WARNING)

MAP_FIELDS = ["pdb", "seed", "sample", "chain_mapping", "anchor_seed", "anchor_sample", "seconds"]
ROLE_FIELDS = ["pdb", "chain_1", "chain_2", "ab_chain", "ag_chain"]


def pool_rows(pdbs=None):
    """{pdb: {(seed, sample): {iface_key: pair_iptm}}} from the pool CSV."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(str(paths.PXM2025_POOL_CSV)) as fh:
        for r in csv.DictReader(fh):
            if pdbs and r["pdb"] not in pdbs:
                continue
            # the pool CSV names these chain1/chain2; the roles CSV names them chain_1/chain_2
            out[r["pdb"]][(r["seed"], int(r["sample"]))][f"{r['chain1']}{r['chain2']}"] = float(
                r["pair_iptm"])
    return out


def pick_anchor(poses):
    """(seed, sample) of the pose with the highest MEAN pair_iptm across the complex's interfaces.

    Poses missing from some interface are averaged over the interfaces they do appear in; requiring
    presence everywhere would exclude exactly the poses that bind an unusual copy, which are the ones
    the diversity walk is meant to surface.
    """
    return max(poses, key=lambda k: sum(poses[k].values()) / max(len(poses[k]), 1))


def one(task):
    """Map one pose onto its complex's anchor. Row format matches chain_map.py's."""
    pdb, seed, sample, path, aseed, asamp, apath = task
    from pxmeter.mapping import MappingResult
    t = time.time()
    base = {"pdb": pdb, "seed": seed, "sample": sample,
            "anchor_seed": aseed, "anchor_sample": asamp}
    if (seed, sample) == (aseed, asamp):
        # the anchor against itself: identity, and asking the mapper for it invites a needless failure
        import gemmi
        st = gemmi.read_structure(apath)
        st.setup_entities()
        cm = {c.name: c.name for c in st[0]}
    else:
        try:
            m = MappingResult.from_cifs(apath, path)   # (reference, model) -> ref_to_model
        except Exception as e:
            return {**base, "chain_mapping": "", "seconds": round(time.time() - t, 2),
                    "anchor_sample": f"FAILED: {type(e).__name__}"}
        cm = {str(k): str(v) for k, v in m.chain_mapping.items()}
    return {**base, "chain_mapping": json.dumps(cm, sort_keys=True),
            "seconds": round(time.time() - t, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pdbs", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    A = ap.parse_args()

    want = set(A.pdbs.split(",")) if A.pdbs else None
    pool = pool_rows(want)
    os.makedirs(A.out_dir, exist_ok=True)
    fmap, frole = f"{A.out_dir}/chain_maps_anchor.csv", f"{A.out_dir}/chain_roles_anchor.csv"

    # crystal roles + the anchor's own crystal mapping, used ONLY to name the anchor's chain pairs
    sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")
    import metrics
    xtal_roles, xtal_maps = metrics.load_roles(), metrics.load_chain_maps()

    tasks, roles, skipped = [], [], []
    for pdb, poses in sorted(pool.items()):
        aseed, asamp = pick_anchor(poses)
        apath = paths.pool_pose(pdb, aseed, asamp)
        am = xtal_maps.get((pdb, aseed, asamp))
        if not os.path.exists(apath) or not am:
            skipped.append(f"{pdb} (anchor pose has no cif or no crystal mapping)")
            continue
        for key in sorted(poses):
            p = paths.pool_pose(pdb, key[0], key[1])
            if os.path.exists(p):
                tasks.append((pdb, key[0], key[1], p, aseed, asamp, apath))
        for ik in sorted(poses[(aseed, asamp)] if (aseed, asamp) in poses else {}):
            iface = f"{pdb}_{ik}"
            if iface not in xtal_roles:
                continue
            ab_ref, ag_ref = xtal_roles[iface]
            if ab_ref in am and ag_ref in am:
                roles.append([pdb, ik[0], ik[1], am[ab_ref], am[ag_ref]])
    for s in skipped:
        print(f"  SKIPPED {s}")

    with open(frole, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ROLE_FIELDS)
        w.writerows(roles)
    print(f"  interfaces with anchor-frame roles: {len(roles)}  ->  {frole}")

    print(f"  poses to map: {len(tasks)}  jobs: {A.jobs}", flush=True)
    t0, nfail = time.time(), 0
    with open(fmap, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MAP_FIELDS)
        w.writeheader()
        import multiprocessing as mp
        with mp.Pool(A.jobs) as pool_:
            for i, row in enumerate(pool_.imap_unordered(one, tasks, chunksize=8), 1):
                w.writerow(row)
                fh.flush()
                nfail += not row["chain_mapping"]
                if i % 500 == 0 or i == len(tasks):
                    el = time.time() - t0
                    print(f"  {i}/{len(tasks)}  {el/i:.3f}s/pose  "
                          f"eta {(len(tasks)-i)*el/i/60:.0f}m  failed {nfail}", flush=True)
    print(f"  wrote {fmap}  ({len(tasks)} poses, {nfail} failed, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
