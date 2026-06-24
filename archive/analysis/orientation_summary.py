"""Prelim orientation findings across several antibody-antigen interfaces.

GT-free: for each interface we superpose every seed's antibody (the interface's
antibody chain) onto a reference good seed, carry the antigen into that frame, and
ask:
  - fold or orientation problem?  -> antigen MONOMER RMSD spread (low = folds
    consistent) vs antigen POSE RMSD spread (high = orientation varies).
  - is there a dominant decoy?     -> cluster the antigen poses (connected
    components at 10 A) and report sizes + mean DockQ.
  - how wrong is the decoy?        -> rigid rotation of the dominant-decoy antigen
    vs the (good-seed) native proxy.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_7txw_poses import parse_cif_ca, kabsch, common  # noqa: E402
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import ESM_PREDICTIONS, TARGETS_CSV, CSV_DIR  # noqa: E402

DOCKQ = CSV_DIR / "dockq_ost_esmfold2fast_tts_all_samples.csv"
RUN = "esmfold2fast_tts"
GOOD_T, BAD_T, CLUST_T = 0.5, 0.23, 10.0

INTERFACES = [
    ("7txw", "A", "C"), ("8vui", "B", "C"), ("7yxu", "A", "C"),
    ("8tmz", "B", "C"), ("9au1", "C", "E"), ("8ido", "A", "D"), ("8fdd", "B", "C"),
]


def targets_map():
    out = {}
    with open(TARGETS_CSV) as f:
        for r in csv.DictReader(f):
            k = (r["pdb_id"].split("-")[0],
                 frozenset({r["interface_chain_id_1"], r["interface_chain_id_2"]}))
            out[k] = (r["antibody_chain"], r["antigen_chain"])
    return out


def best_dockq(pdb, c1, c2):
    out = {}
    with open(DOCKQ) as f:
        for r in csv.DictReader(f):
            if r["assembly_id"].split("-")[0] != pdb or {r["chain1"], r["chain2"]} != {c1, c2}:
                continue
            try:
                dq = float(r["dockq"])
            except ValueError:
                continue
            out[r["seed"]] = max(out.get(r["seed"], 0.0), dq)
    return out


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def components(D, thr):
    n = len(D); seen = [False] * n; comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []; seen[i] = True
        while stack:
            u = stack.pop(); comp.append(u)
            for v in range(n):
                if not seen[v] and D[u, v] < thr:
                    seen[v] = True; stack.append(v)
        comps.append(comp)
    return comps


def analyze(pdb, c1, c2, tgt):
    ab_ag = tgt.get((pdb, frozenset({c1, c2})))
    if not ab_ag:
        return f"{pdb} {c1}|{c2}: not in targets"
    _, ag_chain = ab_ag
    ab_iface = c1 if c2 == ag_chain else c2
    dq = best_dockq(pdb, c1, c2)
    good = sorted(int(s) for s, v in dq.items() if s != "baseline_seed" and v >= GOOD_T)
    bad = [s for s, v in dq.items() if s != "baseline_seed" and v < BAD_T]
    if not good:
        return f"{pdb} {c1}|{c2}: no good seeds"
    pred = ESM_PREDICTIONS / RUN / pdb
    ref_seed = max(good, key=lambda s: dq.get(str(s), 0))
    ref = parse_cif_ca(pred / f"seed_{ref_seed}" / "predictions" / f"{pdb}_sample_0.cif")
    if ab_iface not in ref or ag_chain not in ref:
        return f"{pdb} {c1}|{c2}: ref missing chains {ab_iface}/{ag_chain}"
    ag_keys = sorted(ref[ag_chain])

    labels, coords, mono = [], [], []
    for d in sorted(pred.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
        s = int(d.name.split("_")[1])
        f = d / "predictions" / f"{pdb}_sample_0.cif"
        if not f.exists():
            continue
        st = parse_cif_ca(f)
        if ab_iface not in st or ag_chain not in st:
            continue
        P, Q, _ = common(st, ref, ab_iface)
        if len(P) < 3:
            continue
        R, t = kabsch(P, Q)
        keys = [k for k in ag_keys if k in st[ag_chain]]
        labels.append(s)
        coords.append((keys, np.array([R @ st[ag_chain][k] + t for k in keys])))
        Pa = np.array([st[ag_chain][k] for k in keys])
        Qa = np.array([ref[ag_chain][k] for k in keys])
        Rr, tr = kabsch(Pa, Qa)
        mono.append(np.sqrt(((Pa @ Rr.T + tr - Qa) ** 2).sum(1).mean()))

    n = len(labels)
    dicts = [dict(zip(k, a)) for k, a in coords]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ck = [k for k in dicts[i] if k in dicts[j]]
            a = np.array([dicts[i][k] for k in ck]); b = np.array([dicts[j][k] for k in ck])
            D[i, j] = D[j, i] = np.sqrt(((a - b) ** 2).sum(1).mean())

    pose_spread = np.median(D[np.triu_indices(n, 1)])
    comps = components(D, CLUST_T)
    comps.sort(key=len, reverse=True)
    idx_ref = labels.index(ref_seed)
    nat = next(c for c in comps if idx_ref in c)
    decoys = [c for c in comps if c is not nat]
    dom = max(decoys, key=len) if decoys else []

    def mean_dq(comp):
        v = [dq.get(str(labels[i]), float("nan")) for i in comp]
        return np.nanmean(v)

    # rotation of dominant decoy antigen vs native proxy (both in ref frame)
    rot = float("nan")
    if dom:
        rep = dom[0]
        ck = [k for k in dicts[idx_ref] if k in dicts[rep]]
        a = np.array([dicts[idx_ref][k] for k in ck]); b = np.array([dicts[rep][k] for k in ck])
        Rr, _ = kabsch(a, b)
        rot = angle(Rr)

    pep = " (PEPTIDE!)" if len(ag_keys) < 30 else ""
    lines = [
        f"### {pdb} {c1}|{c2}   antigen=chain {ag_chain} ({len(ag_keys)} res){pep}, "
        f"antibody-iface=chain {ab_iface}",
        f"  seeds={n}  good={len(good)} bad={len(bad)}",
        f"  antigen MONOMER RMSD (fold consistency): median {np.median(mono):.2f} A  "
        f"max {np.max(mono):.2f} A   -> {'folds consistent (orientation problem)' if np.median(mono) < 2 else 'folds VARY too'}",
        f"  antigen POSE RMSD spread (orientation variability): median {pose_spread:.1f} A",
        f"  clusters at <{CLUST_T:.0f}A: " + ", ".join(
            f"n={len(c)}(dq{mean_dq(c):.2f}{',NATIVE' if c is nat else ''})" for c in comps[:6]),
        f"  native cluster: n={len(nat)} (dq {mean_dq(nat):.2f})   "
        f"dominant decoy: n={len(dom)} (dq {mean_dq(dom):.2f})  rotation vs native: {rot:.0f} deg"
        if dom else f"  native cluster: n={len(nat)} (dq {mean_dq(nat):.2f})  [no decoy cluster]",
    ]
    return "\n".join(lines)


def main():
    tgt = targets_map()
    for pdb, c1, c2 in INTERFACES:
        print(analyze(pdb, c1, c2, tgt))
        print()


if __name__ == "__main__":
    main()
