#!/usr/bin/env python
"""Name every pose's chains after one chosen pose, so "interface A-B" means the same thing in all of them.

    anchor_relabel.py --out-dir DIR [--pdbs 8ror,9puo] [--jobs 8]

WHEN THIS DOES ANYTHING. Only when a complex contains more than one copy of an entity. With one
antibody and one antigen there is a single possible chain assignment -- the anchor's A can only match
the pose's A, since that is the only chain with the antibody's sequence -- so the mapping is the
identity and this step is a no-op. Of the 131 benchmark complexes, 52 have no duplication at all, 44
duplicate the antibody, and 35 duplicate both; the largest, 9jo7, has 5 antibody and 5 antigen chains.
Measured over the 73 multi-interface complexes: 24 never produce a non-identity mapping, 49 do, and 42%
of all poses get at least one chain renamed.

WHY THE LABEL MATTERS EVEN THOUGH THE COPIES ARE IDENTICAL. The copies have the same sequence but not
the same position, and in any given pose the antibody touches one of them and not the other:

    pose 1: antibody A binds the copy labelled B  ->  (A,B) is a real interface, (A,C) is nothing
    pose 2: antibody A binds the copy labelled C  ->  (A,C) is a real interface, (A,B) is nothing

Ranking poses by "ipTM at A-B" then compares pose 1's docking site against a pair of chains in pose 2
that never touch. The diversity filter has the same problem: pose-pose DockQ at A-B scores pose 1's
interface against pose 2's non-interface and returns near zero, so two poses that dock identically look
completely different -- purely because the copies were named in a different order.

Protenix emits chains in a fixed order from the input, but which physical copy lands in which slot
depends on the fold, so two runs of one complex can put the same copy at B or at C.

ONE ANCHOR PER COMPLEX, not per interface. The mapping pairs each anchor chain with exactly one pose
chain and no two share a partner, so the anchor's (A,B) and (A,C) necessarily land on different chain
pairs of the pose. Two independent per-interface mappings could both pick the pose's (A,B) pair, putting
one physical interface into two candidate lists.

OUTPUT, both drop-in for build_matrices.py --maps/--roles:

    chain_maps_anchor.csv    per pose: {anchor_chain: pose_chain}
    chain_roles_anchor.csv   per interface: which ANCHOR chains are the antibody and the antigen

FUNCTIONS

    pool_rows()    pool csv -> {pdb: {(seed, sample): {interface: pair_iptm}}}
    pick_anchor()  the poses of one complex -> the (seed, sample) to use as reference
    map_to_anchor()  one pose -> its chain mapping onto the anchor, as a csv row
    anchor_roles()   the anchor's chain pair per interface, for the roles csv
    main()         pick an anchor per complex, map every pose, write both csvs

The crystal is read for one thing only: naming. xtal_roles/xtal_maps decide which of the anchor's chain
pairs is called "8w5d_AB", so the output can be compared against the crystal-framed run. No donor
depends on it. pick_anchor also averages ipTM over the reference interface list, which is a real
dependency -- averaging over all antibody x antigen pairs instead would remove it.
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
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")

import metrics  # noqa: E402
import paths  # noqa: E402

logging.disable(logging.WARNING)   # pxmeter logs a warning per unmapped ligand, thousands of poses

MAP_FIELDS = ["pdb", "seed", "sample", "chain_mapping", "anchor_seed", "anchor_sample", "seconds"]
ROLE_FIELDS = ["pdb", "chain_1", "chain_2", "ab_chain", "ag_chain"]


def pool_rows(pdbs=None):
    """in: optional subset of complexes.  out: {pdb: {(seed, sample): {interface: pair_iptm}}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(str(paths.PXM2025_POOL_CSV)) as fh:
        for r in csv.DictReader(fh):
            if pdbs and r["pdb"] not in pdbs:
                continue
            # pool csv spells these chain1/chain2; the roles csv spells them chain_1/chain_2
            key = f"{r['chain1']}{r['chain2']}"
            out[r["pdb"]][(r["seed"], int(r["sample"]))][key] = float(r["pair_iptm"])
    return out


def pick_anchor(poses):
    """in: {(seed, sample): {interface: pair_iptm}}.  out: the (seed, sample) with the highest mean.

    Averaged over the interfaces a pose appears in, not over all of them: requiring presence
    everywhere would exclude the poses that bind an unusual copy, which are the ones worth surfacing.
    """
    return max(poses, key=lambda k: sum(poses[k].values()) / max(len(poses[k]), 1))


def map_to_anchor(task):
    """in: (pdb, seed, sample, pose path, anchor seed, anchor sample, anchor path)
    out: a csv row; chain_mapping is "" and anchor_sample carries the reason when mapping fails
    """
    pdb, seed, sample, path, aseed, asamp, apath = task
    from pxmeter.mapping import MappingResult
    t = time.time()
    base = {"pdb": pdb, "seed": seed, "sample": sample,
            "anchor_seed": aseed, "anchor_sample": asamp}
    if (seed, sample) == (aseed, asamp):
        import gemmi                              # anchor against itself is the identity mapping
        st = gemmi.read_structure(apath)
        st.setup_entities()
        cm = {c.name: c.name for c in st[0]}
    else:
        try:
            m = MappingResult.from_cifs(apath, path)      # (reference, model) -> ref_to_model
        except Exception as e:
            return {**base, "chain_mapping": "", "seconds": round(time.time() - t, 2),
                    "anchor_sample": f"FAILED: {type(e).__name__}"}
        cm = {str(k): str(v) for k, v in m.chain_mapping.items()}
    return {**base, "chain_mapping": json.dumps(cm, sort_keys=True),
            "seconds": round(time.time() - t, 2)}


def anchor_roles(pdb, interfaces, anchor_xtal_map, xtal_roles):
    """in: the anchor's crystal mapping and the reference roles.
    out: rows of [pdb, c1, c2, anchor ab chain, anchor ag chain]
    """
    rows = []
    for ik in sorted(interfaces):
        iface = f"{pdb}_{ik}"
        if iface not in xtal_roles:
            continue
        ab_ref, ag_ref = xtal_roles[iface]
        if ab_ref in anchor_xtal_map and ag_ref in anchor_xtal_map:
            rows.append([pdb, ik[0], ik[1], anchor_xtal_map[ab_ref], anchor_xtal_map[ag_ref]])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pdbs", default=None, help="comma-separated subset; default is every complex")
    ap.add_argument("--jobs", type=int, default=8)
    A = ap.parse_args()

    pool = pool_rows(set(A.pdbs.split(",")) if A.pdbs else None)
    os.makedirs(A.out_dir, exist_ok=True)
    fmap = f"{A.out_dir}/chain_maps_anchor.csv"
    frole = f"{A.out_dir}/chain_roles_anchor.csv"
    xtal_roles, xtal_maps = metrics.load_roles(), metrics.load_chain_maps()

    tasks, roles, skipped = [], [], []
    for pdb, poses in sorted(pool.items()):
        aseed, asamp = pick_anchor(poses)
        apath = paths.pool_pose(pdb, aseed, asamp)
        anchor_xtal_map = xtal_maps.get((pdb, aseed, asamp))
        if not os.path.exists(apath) or not anchor_xtal_map:
            skipped.append(f"{pdb} (anchor pose has no cif or no crystal mapping)")
            continue
        for seed, sample in sorted(poses):
            p = paths.pool_pose(pdb, seed, sample)
            if os.path.exists(p):
                tasks.append((pdb, seed, sample, p, aseed, asamp, apath))
        roles += anchor_roles(pdb, poses.get((aseed, asamp), {}), anchor_xtal_map, xtal_roles)
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
        with mp.Pool(A.jobs) as workers:
            for i, row in enumerate(workers.imap_unordered(map_to_anchor, tasks, chunksize=8), 1):
                w.writerow(row)
                fh.flush()          # a kill loses the pose in flight, not the run
                nfail += not row["chain_mapping"]
                if i % 500 == 0 or i == len(tasks):
                    el = time.time() - t0
                    print(f"  {i}/{len(tasks)}  {el/i:.3f}s/pose  "
                          f"eta {(len(tasks)-i)*el/i/60:.0f}m  failed {nfail}", flush=True)
    print(f"  wrote {fmap}  ({len(tasks)} poses, {nfail} failed, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
