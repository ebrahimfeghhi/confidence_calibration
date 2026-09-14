#!/usr/bin/env python
"""Pick the N seeds whose epitopes are furthest apart, per complex.

For the trajectory experiment we want seeds that demonstrably land in DIFFERENT places, so that
watching their per-recycle epitopes tells us whether trajectories hop between sites or commit early.
Greedy farthest-point traversal on epitope Jaccard distance: start from the pair furthest apart, then
repeatedly add the seed whose minimum distance to everything already chosen is largest.

A seed is represented by the UNION of its 5 poses' epitopes -- the territory that seed covered.

Seeds are a property of the COMPLEX (one inference run yields every interface), so selection is done
per complex using its highest-spread interface, and both interfaces of a complex reuse the same run.

N=8 rather than 5 on purpose: Protenix is not bit-reproducible end to end, so a re-run of a seed need
not land where it did before. Picking 8 and keeping the 5 most diverse AFTERWARDS guarantees the
analysis set is actually spread out.
"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, "scripts")
from cdr_h3_rmsd import poses_free

N_PICK = 8


def seed_sets(iface):
    z = np.load(f"artifacts/distmat/epitope_distmat_{iface}.npz", allow_pickle=True)
    keep = np.array([bool(x) for x in z["sets"]])
    pdb = iface.rsplit("_", 1)[0]
    ps = [(s, k) for _, s, k, _ in poses_free(pdb) if 101 <= s <= 150]
    if len(ps) != len(keep):
        return None
    d = collections.defaultdict(set)
    for (s, k), v, kp in zip(ps, z["sets"], keep):
        if kp:
            d[s] |= {int(x) for x in v.split(",")}
    return {s: v for s, v in d.items() if v}


def farthest_point(sets, n):
    seeds = sorted(sets)
    J = lambda a, b: 1 - len(a & b) / len(a | b) if (a | b) else 0.0
    D = np.array([[J(sets[a], sets[b]) for b in seeds] for a in seeds])
    i, j = np.unravel_index(np.argmax(D), D.shape)
    chosen = [i, j]
    while len(chosen) < min(n, len(seeds)):
        rest = [x for x in range(len(seeds)) if x not in chosen]
        nxt = max(rest, key=lambda x: min(D[x, c] for c in chosen))
        chosen.append(nxt)
    return [seeds[c] for c in chosen], float(np.mean([D[a, b] for a in chosen for b in chosen if a != b]))


def main():
    ifaces = json.load(open("artifacts/traj20.json"))
    spread = {}
    for i in ifaces:
        z = np.load(f"artifacts/distmat/epitope_distmat_{i}.npz", allow_pickle=True)
        keep = np.array([bool(x) for x in z["sets"]])
        D = z["D"][np.ix_(keep, keep)]
        iu = np.triu_indices(D.shape[0], 1)
        spread[i] = float(D[iu].mean())
    by_pdb = collections.defaultdict(list)
    for i in ifaces:
        by_pdb[i.rsplit("_", 1)[0]].append(i)
    out = {}
    print(f'{"complex":<8}{"driver interface":<14}{"seeds chosen":<46}{"mean pairwise dist":>19}')
    for pdb, its in sorted(by_pdb.items()):
        drv = max(its, key=lambda x: spread[x])
        ss = seed_sets(drv)
        if not ss or len(ss) < N_PICK:
            print(f"{pdb:<8}{drv:<14}SKIP (only {len(ss) if ss else 0} usable seeds)")
            continue
        picked, md = farthest_point(ss, N_PICK)
        out[pdb] = dict(seeds=sorted(picked), driver=drv, interfaces=its, mean_dist=md)
        print(f'{pdb:<8}{drv:<14}{",".join(str(s) for s in sorted(picked)):<46}{md:>19.2f}')
    json.dump(out, open("artifacts/traj20_seeds.json", "w"), indent=1)
    print(f"\n{len(out)} complexes -> artifacts/traj20_seeds.json")
    print(f"total decodes: {len(out)*N_PICK*10} (8 seeds x 10 recycles)")


if __name__ == "__main__":
    main()
