#!/usr/bin/env python
"""What does seed scaling actually buy, under a fair single-seed baseline?

THE PROTOCOL (the standard for ab-ag analyses from here on):
  1-seed condition  -- within ONE seed, take the pose its own ipTM ranks first among that seed's 5
                       diffusion samples. Do this for all 50 seeds, giving 50 candidate outcomes, and
                       report the one with the MEDIAN DockQ. Picking a single arbitrary seed would
                       make the baseline a coin flip; averaging DockQ would give a number with no
                       pose attached, and every metric here has to be measured ON a pose.
                       With 50 values the median lies between the 25th and 26th, so the LOWER median
                       is used -- a real pose, so its metrics are measurements and not interpolations.
  50-seed condition -- the pose ipTM ranks first pooled over all 250.

Both conditions are ipTM-selected, so the comparison isolates what extra SAMPLING buys and never
credits scaling with an oracle it would not have at inference time.

METRICS, on each selected pose: epitope recall/jaccard (did it find the right antigen surface),
paratope recall/jaccard (did it use the right antibody face), and fnat (did it recover the actual
native contact pairs). Epitope and paratope overlap are NOT DockQ components, so unlike fnat they can
be compared against DockQ without circularity.

ipTM is the INTERFACE value chain_pair_iptm[mdl_ab][mdl_ag], never the global `iptm`, which blends in
intra-antibody confidence and hides the interface signal. Chains come from scripts/iface_chains.py.

Usage: /home/ebrahim/miniconda3/envs/foldbench/bin/python scripts/selection_protocol.py
Outputs artifacts/abag_selection_protocol.csv (one row per interface, both conditions).
"""
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iface_chains import resolve  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
SEED_LO, SEED_HI = 101, 150
METRICS = ["dockq", "fnat", "epi_recall", "epi_jac", "para_recall", "para_jac"]


def chain_idx(ch):
    base, alpha = ch.split("-")[0], ""
    for c in base:
        if not c.isalpha():
            break
        alpha += c
    idx = 0
    for c in alpha:
        idx = idx * 26 + (ord(c.upper()) - ord("A") + 1)
    return idx - 1


def load_overlap(fn, rec, jac):
    out = {}
    for r in csv.DictReader(open(f"{EXP}/artifacts/{fn}")):
        if SEED_LO <= int(r["seed"]) <= SEED_HI:
            out[(r["pdb"], r["ab_chain"], r["ag_chain"], r["seed"], r["sample"])] = \
                (float(r[rec]), float(r[jac]))
    return out


_conf_cache = {}


def pair_iptm(pdb, seed, smp, m_ab, m_ag):
    """Interface ipTM for THIS pose, from chain_pair_iptm over the model chains OST matched."""
    key = (pdb, seed, smp)
    if key not in _conf_cache:
        f = f"{PRED}/{pdb}/seed_{seed}/predictions/{pdb}_summary_confidence_sample_{smp}.json"
        try:
            _conf_cache[key] = json.load(open(f)).get("chain_pair_iptm")
        except Exception:
            _conf_cache[key] = None
    M = _conf_cache[key]
    if not M:
        return None
    i, j = chain_idx(m_ab), chain_idx(m_ag)
    if 0 <= i < len(M) and 0 <= j < len(M[i]):
        return M[i][j]
    return None


def load_rmsd(fn, a, b, key3=True):
    """-> {(pdb, ab, ag, seed, sample): (local, global)}.

    key3=False for CDR H3, whose table is keyed by ANTIBODY CHAIN ONLY (pdb, chain, seed, sample) --
    H3 is a property of the heavy chain, not of an interface, so a light-chain interface row has no
    H3 value and is left missing rather than silently paired with the heavy chain's number.
    """
    out = {}
    for r in csv.DictReader(open(f"{EXP}/artifacts/{fn}")):
        if not (SEED_LO <= int(r["seed"]) <= SEED_HI):
            continue
        k = ((r["pdb"], r["ab_chain"], r["ag_chain"], r["seed"], r["sample"]) if key3
             else (r["pdb"], r["chain"], r["seed"], r["sample"]))
        try:
            out[k] = (float(r[a]), float(r[b]))
        except (ValueError, KeyError):
            pass
    return out


def main():
    epi = load_overlap("epitope_overlap.csv", "recall", "jaccard")
    par = load_overlap("paratope_overlap.csv", "recall", "jaccard")
    prm = load_rmsd("paratope_rmsd.csv", "para_local", "para_global")
    h3 = load_rmsd("cdr_h3_rmsd_all.csv", "h3_local", "h3_global", key3=False)
    # TWO epitope-RMSD tables exist and disagree on 13,057 of 50,575 shared poses, with no script,
    # doc or commit explaining what v2 changed (artifacts are gitignored; only ops/epitope_rmsd_v2.log
    # survives, showing v2 added validation and dropped some poses). Rather than guess which is
    # canonical, BOTH are carried through so any conclusion can be checked against each.
    erm = load_rmsd("epitope_rmsd.csv", "epi_local", "epi_global")
    erm2 = load_rmsd("epitope_rmsd_v2.csv", "epi_local", "epi_global")

    poses = collections.defaultdict(list)
    keys = sorted(set(epi) & set(par))
    for k in keys:
        pdb, ab, ag, seed, smp = k
        try:
            ch = resolve(pdb, seed, smp, ab, ag)
        except (KeyError, FileNotFoundError):
            continue
        if ch["dockq"] is None or ch["fnat"] is None:
            continue
        pi = pair_iptm(pdb, seed, smp, ch["mdl_ab"], ch["mdl_ag"])
        if pi is None:
            continue
        poses[f"{pdb}_{ab}{ag}"].append(dict(
            seed=int(seed), sample=int(smp), iptm=pi, dockq=ch["dockq"], fnat=ch["fnat"],
            epi_recall=epi[k][0], epi_jac=epi[k][1],
            para_recall=par[k][0], para_jac=par[k][1],
            para_local=prm.get(k, (None, None))[0], para_global=prm.get(k, (None, None))[1],
            epirm_local=erm.get(k, (None, None))[0], epirm_global=erm.get(k, (None, None))[1],
            epirm2_local=erm2.get(k, (None, None))[0], epirm2_global=erm2.get(k, (None, None))[1],
            h3_local=h3.get((pdb, ab, seed, smp), (None, None))[0],
            h3_global=h3.get((pdb, ab, seed, smp), (None, None))[1]))

    rows = []
    for iface, ps in sorted(poses.items()):
        by_seed = collections.defaultdict(list)
        for p in ps:
            by_seed[p["seed"]].append(p)
        if len(by_seed) < 25 or len(ps) < 100:
            continue
        # 1-seed: ipTM's pick within each seed, then the MEDIAN-DockQ one of those (lower median)
        per_seed = [max(v, key=lambda x: x["iptm"]) for v in by_seed.values()]
        per_seed.sort(key=lambda x: x["dockq"])
        one = per_seed[(len(per_seed) - 1) // 2]
        # 50-seed: ipTM's pick pooled over everything
        allsel = max(ps, key=lambda x: x["iptm"])
        # ORACLE arm, same shape: within a seed take its BEST DockQ (not ipTM's pick), then the
        # median seed. Upper bound on what one seed could give if selection were perfect.
        best_per_seed = sorted(max(p["dockq"] for p in v) for v in by_seed.values())
        # ORACLE EPITOPE arm: same shape but on epitope jaccard, to separate COVERAGE (did scaling
        # put a right-surface pose anywhere in the pool?) from SELECTION (did ipTM then pick it?).
        best_epi_seed = sorted(max(p["epi_jac"] for p in v) for v in by_seed.values())
        # The PARATOPE oracle, built identically, so the epitope and paratope oracle regressions are
        # the same estimator and any difference between them is about the quantity, not the method.
        best_para_seed = sorted(max(p["para_jac"] for p in v) for v in by_seed.values())
        # Per-axis 1-seed baselines: the median of the SAME quantity being compared. sel1_* above
        # takes the DockQ-median pose, so its epitope Jaccard is that pose's incidental value; for a
        # gain-in-epitope axis the honest baseline is the median epitope Jaccard among the 50 ipTM
        # picks. The two baselines are therefore different poses -- intentional.
        med = lambda xs: sorted(xs)[(len(xs) - 1) // 2]
        medn = lambda key: (med([q[key] for q in per_seed if q[key] is not None])
                            if any(q[key] is not None for q in per_seed) else None)
        row = dict(interface=iface, n_seeds=len(by_seed), n_poses=len(ps),
                   oracle1_dockq=round(best_per_seed[(len(best_per_seed) - 1) // 2], 4),
                   oracle_dockq=max(p["dockq"] for p in ps),
                   oracle1_epi_jac=round(best_epi_seed[(len(best_epi_seed) - 1) // 2], 4),
                   oracle_epi_jac=round(max(p["epi_jac"] for p in ps), 4),
                   oracle1_para_jac=round(best_para_seed[(len(best_para_seed) - 1) // 2], 4),
                   oracle_para_jac=round(max(p["para_jac"] for p in ps), 4),
                   sel1med_epi_jac=round(med([p["epi_jac"] for p in per_seed]), 4),
                   sel1med_epi_recall=round(med([p["epi_recall"] for p in per_seed]), 4),
                   sel1med_para_jac=round(med([p["para_jac"] for p in per_seed]), 4),
                   sel1med_para_local=medn("para_local"), sel1med_para_global=medn("para_global"),
                   sel1med_h3_local=medn("h3_local"), sel1med_h3_global=medn("h3_global"),
                   sel1med_epirm_local=medn("epirm_local"),
                   sel1med_epirm_global=medn("epirm_global"),
                   sel1med_epirm2_local=medn("epirm2_local"),
                   sel1med_epirm2_global=medn("epirm2_global"))
        for tag, p in (("sel1", one), ("sel50", allsel)):
            for m in ("para_local", "para_global", "h3_local", "h3_global",
                      "epirm_local", "epirm_global", "epirm2_local", "epirm2_global"):
                row[f"{tag}_{m}"] = p[m]
        for tag, p in (("sel1", one), ("sel50", allsel)):
            row[f"{tag}_seed"], row[f"{tag}_sample"] = p["seed"], p["sample"]
            row[f"{tag}_iptm"] = round(p["iptm"], 4)
            for m in METRICS:
                row[f"{tag}_{m}"] = round(p[m], 4)
        rows.append(row)

    out = f"{EXP}/artifacts/abag_selection_protocol.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} interfaces (>=25 seeds, >=100 poses)\n")
    print("mean +/- SD across interfaces. `% improvement` is computed PER INTERFACE as")
    print("100*(50seed - 1seed)/1seed, then averaged -- interfaces with a zero 1-seed baseline are")
    print("undefined and dropped, so that column carries its own N.\n")
    print(f"{'metric':<14}{'N':>5}{'1 seed':>16}{'50 seeds':>16}{'delta':>16}"
          f"{'% improvement':>22}{'n%':>5}{'improved':>10}")
    for m in METRICS:
        a = np.array([r[f"sel1_{m}"] for r in rows], float)
        b = np.array([r[f"sel50_{m}"] for r in rows], float)
        d = b - a
        pct = np.array([100 * (y - x) / x for x, y in zip(a, b) if x > 1e-9])
        print(f"{m:<14}{len(a):>5}{a.mean():>9.3f} +/-{a.std():<5.3f}{b.mean():>9.3f} +/-{b.std():<5.3f}"
              f"{d.mean():>+9.3f} +/-{d.std():<5.3f}{pct.mean():>+13.1f}% +/-{pct.std():<7.1f}"
              f"{len(pct):>5}{100 * (d > 0).mean():>9.0f}%")

    a = np.array([r["sel1_dockq"] for r in rows]); b = np.array([r["sel50_dockq"] for r in rows])
    orc = np.array([r["oracle_dockq"] for r in rows])
    print(f"\n{'acceptable (DockQ>=0.23)':<28} 1 seed {100 * (a >= .23).mean():.1f}%"
          f"   50 seeds {100 * (b >= .23).mean():.1f}%   oracle {100 * (orc >= .23).mean():.1f}%")
    print(f"{'mean oracle DockQ':<28} {orc.mean():.3f}  "
          f"(50-seed selection captures {100 * (b.mean() - a.mean()) / (orc.mean() - a.mean()):.0f}%"
          f" of the headroom)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
