#!/usr/bin/env python
"""How much NEW territory does each extra seed buy? -- the redundancy behind log-linear scaling.

Unconditional seed scaling draws i.i.d. from a fixed distribution, so most draws land where earlier
ones already went. This measures that waste directly, without any threshold or clustering.

RAREFACTION. For an interface, sample n seeds at random, take the UNION of all antigen/partner
residues its poses contact, average over many draws. Union size grows like  U(n) = k * n^alpha:

    alpha = 1  every seed brings entirely fresh territory (no redundancy)
    alpha = 0  every seed lands in the same place (total redundancy)

alpha is therefore the redundancy number, and a small alpha is exactly what produces a log-linear
success curve. Also reported: seeds needed to reach 90% of the coverage the full pool achieves.

Threshold-free by construction -- no cluster count, no jaccard cutoff, the two things that have
repeatedly turned out to drive results here.

Usage: .../python scripts/seed_redundancy.py            (both benchmarks)
"""
import collections, csv, glob, os, re, sys
import numpy as np
sys.path.insert(0, "scripts")

RNG = np.random.default_rng(0)
NDRAW = 40                      # random subsets averaged at each n


def rarefy(seed_sets, ns):
    """seed_sets: {seed: set(residues)} -> mean union size at each n in ns."""
    seeds = list(seed_sets)
    out = []
    for n in ns:
        if n > len(seeds):
            out.append(np.nan); continue
        v = []
        for _ in range(NDRAW):
            pick = RNG.choice(len(seeds), size=n, replace=False)
            u = set()
            for i in pick:
                u |= seed_sets[seeds[i]]
            v.append(len(u))
        out.append(float(np.mean(v)))
    return np.array(out)


def alpha_of(ns, u):
    """slope of log(union) vs log(n) -- the growth exponent."""
    m = ~np.isnan(u) & (u > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(np.array(ns)[m]), np.log(u[m]), 1)[0])


def load_abag():
    """{interface: {seed: set(antigen residues contacted)}} from the corrected epitope dumps."""
    from cdr_h3_rmsd import GT_DIR, _ost, align_map, chain_residues
    from epitope_clusters import contact_residues
    from iface_chains import resolve
    out = collections.defaultdict(lambda: collections.defaultdict(set))
    import gzip, json
    # reuse the per-pose contact sets already computed for the distance matrices
    for f in sorted(glob.glob("artifacts/distmat/epitope_distmat_*.npz")):
        iface = os.path.basename(f)[len("epitope_distmat_"):-4]
        z = np.load(f, allow_pickle=True)
        sets, keep = z["sets"], np.array([bool(x) for x in z["sets"]])
        pdb = iface.rsplit("_", 1)[0]
        from cdr_h3_rmsd import poses_free
        ps = [(s, k) for _, s, k, _ in poses_free(pdb) if 101 <= s <= 150]
        if len(ps) != len(sets):
            continue
        for (s, k), v, kp in zip(ps, sets, keep):
            if kp:
                out[iface][s] |= {int(x) for x in v.split(",")}
    return {k: dict(v) for k, v in out.items() if len(v) >= 40}


def load_pp():
    """{interface: {seed: set(residues contacted on BOTH chains, prefixed by side)}}"""
    res = np.load("artifacts/pp_iface_jaccard.npy", allow_pickle=True).item()
    out = {}
    for iface, r in res.items():
        d = collections.defaultdict(set)
        for seed, smp, s0, s1 in r:
            d[int(seed)] |= {("A", x) for x in s0[0]} | {("B", x) for x in s1[0]}
        if len(d) >= 40:
            out[iface] = dict(d)
    return out


def main():
    NS = [1, 2, 3, 5, 8, 12, 20, 30, 40, 50]
    rows = []
    for lab, data in (("antibody-antigen", load_abag()), ("protein-protein", load_pp())):
        print(f"\n{lab}: {len(data)} interfaces", flush=True)
        for iface, ss in data.items():
            u = rarefy(ss, NS)
            a = alpha_of(NS, u)
            full = len(set().union(*ss.values()))
            need = next((n for n, x in zip(NS, u) if not np.isnan(x) and x >= .9 * full), None)
            rows.append(dict(bench=lab, interface=iface, alpha=a, u1=u[0], u50=u[-1],
                             full=full, n90=need if need else np.nan,
                             ratio=u[-1] / u[0] if u[0] else np.nan))
    with open("artifacts/seed_redundancy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.save("artifacts/seed_redundancy_curves.npy",
            {"ns": NS,
             "abag": {r["interface"]: r for r in rows if r["bench"] == "antibody-antigen"},
             "pp": {r["interface"]: r for r in rows if r["bench"] == "protein-protein"}},
            allow_pickle=True)
    print(f"\n{'benchmark':<20}{'n':>5}{'alpha':>9}{'1 seed':>9}{'50 seeds':>10}{'ratio':>8}{'seeds for 90%':>15}")
    for lab in ("antibody-antigen", "protein-protein"):
        g = [r for r in rows if r["bench"] == lab]
        a = np.array([r["alpha"] for r in g]); rt = np.array([r["ratio"] for r in g])
        n9 = np.array([r["n90"] for r in g], float)
        print(f"{lab:<20}{len(g):>5}{np.median(a):>9.2f}{np.median([r['u1'] for r in g]):>9.0f}"
              f"{np.median([r['u50'] for r in g]):>10.0f}{np.median(rt):>8.1f}"
              f"{np.nanmedian(n9):>15.0f}")
    print("\nalpha = growth exponent of covered residues vs seeds "
          "(1 = no redundancy, 0 = total redundancy)")
    return rows


if __name__ == "__main__":
    main()
