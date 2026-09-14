#!/usr/bin/env python
"""WITHIN each interface, how much of DockQ do epitope and paratope Jaccard explain, jointly?

For ONE interface, take its 250 poses and fit three OLS models across those poses:
    DockQ ~ epitope                     R2_epi
    DockQ ~ paratope                    R2_para
    DockQ ~ epitope + paratope          R2_joint
then decompose, per interface, in the standard way:
    unique epitope  = R2_joint - R2_para
    unique paratope = R2_joint - R2_epi
    shared          = R2_epi + R2_para - R2_joint
Report the DISTRIBUTION of those across interfaces, not a single pooled fit.

WHY WITHIN AND NOT BETWEEN. The r2 = 0.82 result is a BETWEEN-interface regression: one point per
interface, gain-vs-gain, answering "does finding the epitope explain why scaling helps this target".
This asks the complementary question inside a single target -- among these 250 poses, do the good ones
differ from the bad ones in epitope, in paratope, or in both? Between-interface structure cannot leak
in, since every fit uses one interface's own poses.

THE TWO PREDICTORS ARE NOT INDEPENDENT and the decomposition is the point. Paratope Jaccard measures
which ANTIBODY residues are in contact, which changes when the docking site changes, so it is partly
inherited from epitope choice -- measured at r = +0.50 to +0.57 between their per-interface SDs. A
large SHARED term with a near-zero UNIQUE paratope term is therefore the expected signature if the
paratope carries no information of its own.

NEGATIVE UNIQUE TERMS ARE IMPOSSIBLE and are not clipped: R2_joint >= max(R2_epi, R2_para) always, so
any negative value here would be a bug, not a finding. Asserted rather than silently clamped. The
SHARED term legitimately can go negative (suppression) and is left as-is.

DEGENERATE FITS ARE DROPPED, NOT ZEROED: an interface whose DockQ or whose predictors are constant
over all 250 poses has no defined R2. Counted and reported.

EXACT POSE SET enforced per dataset (FoldBench seeds 101-150, PXM 300000-300049, 5 samples each);
an interface missing any pose is dropped and reported.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/within_iface_joint_r2.py
Writes artifacts/within_iface_joint_r2.csv and artifacts/within_iface_joint_r2.png
"""
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
N_SAMPLE = 5
ABAG = lambda r: (r["pdb"], tuple(sorted((r["ab_chain"], r["ag_chain"]))))
PAIR = lambda r: (r["pdb"], tuple(sorted((r["chain1"], r["chain2"]))))

SETS = [
    ("FoldBench", frozenset(range(101, 151)),
     f"{EXP}/artifacts/dockq_per_pose_foldbench.csv", ABAG,
     f"{EXP}/artifacts/epitope_overlap.csv", f"{EXP}/artifacts/paratope_overlap.csv"),
    ("PXM-2025", frozenset(range(300000, 300050)),
     f"{P}/dockq_pxm2025.csv", PAIR,
     f"{P}/epitope_overlap_pxm.csv", f"{P}/paratope_overlap_pxm.csv"),
]


def read(path, key, field, seeds):
    out = {}
    for r in csv.DictReader(open(path)):
        s = int(r["seed"])
        if s in seeds and r.get(field):
            out[(key(r), s, int(r["sample"]))] = float(r[field])
    return out


def r2(y, X):
    """OLS R^2 of y on columns of X (intercept added)."""
    A = np.column_stack([np.ones(len(y))] + [X[:, i] for i in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan


def main():
    rows, dropped = [], collections.Counter()
    for name, seeds, dqp, dqk, epip, parap in SETS:
        dq = read(dqp, dqk, "dockq", seeds)
        epi = read(epip, ABAG, "jaccard", seeds)
        para = read(parap, ABAG, "jaccard", seeds)
        g = collections.defaultdict(list)
        for k in set(dq) & set(epi) & set(para):
            g[k[0]].append((epi[k], para[k], dq[k]))
        for iface, v in g.items():
            if len(v) != len(seeds) * N_SAMPLE:
                dropped[f"{name}:incomplete"] += 1
                continue
            a = np.array(v)
            e, p, y = a[:, 0], a[:, 1], a[:, 2]
            if y.std() < 1e-9 or (e.std() < 1e-9 and p.std() < 1e-9):
                dropped[f"{name}:no variation"] += 1
                continue
            re_ = r2(y, e[:, None]) if e.std() > 1e-9 else 0.0
            rp = r2(y, p[:, None]) if p.std() > 1e-9 else 0.0
            rj = r2(y, np.column_stack([e, p]))
            # R2_joint cannot be below either single-predictor R2; a violation is a bug not a result
            assert rj >= re_ - 1e-8 and rj >= rp - 1e-8, (iface, re_, rp, rj)
            rows.append(dict(dataset=name, interface=f"{iface[0]}_{''.join(iface[1])}",
                             n=len(v), r2_epi=round(re_, 4), r2_para=round(rp, 4),
                             r2_joint=round(rj, 4),
                             uniq_epi=round(rj - rp, 4), uniq_para=round(rj - re_, 4),
                             shared=round(re_ + rp - rj, 4)))

    with open(f"{EXP}/artifacts/within_iface_joint_r2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("median across interfaces, of the WITHIN-interface fit over 250 poses\n")
    print(f"{'dataset':<12}{'n iface':>9}{'R2 epi':>9}{'R2 para':>10}{'R2 joint':>10}"
          f"{'uniq epi':>10}{'uniq para':>11}{'shared':>9}")
    for name, *_ in SETS:
        R = [r for r in rows if r["dataset"] == name]
        m = lambda k: np.median([r[k] for r in R])
        print(f"{name:<12}{len(R):>9}{m('r2_epi'):>9.3f}{m('r2_para'):>10.3f}"
              f"{m('r2_joint'):>10.3f}{m('uniq_epi'):>10.3f}{m('uniq_para'):>11.3f}"
              f"{m('shared'):>9.3f}")
    if dropped:
        print("\ndropped:", dict(dropped))

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    for ax, (name, *_) in zip(axes, SETS):
        R = [r for r in rows if r["dataset"] == name]
        bins = np.linspace(0, 1, 41)
        for k, c, lab in (("r2_epi", "#b3452c", "epitope alone"),
                          ("r2_para", "#31688e", "paratope alone"),
                          ("r2_joint", "#3f7d4e", "both together")):
            v = np.array([r[k] for r in R])
            ax.hist(v, bins=bins, histtype="step", lw=2.0, color=c,
                    label=f"{lab}  (median {np.median(v):.3f})")
        ax.set_title(f"{name}  (n={len(R)} interfaces)", fontsize=13)
        ax.set_xlabel("$R^2$ of DockQ, fit across the interface's 250 poses", fontsize=12)
        ax.set_ylabel("number of interfaces", fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=11, loc="upper left")
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/within_iface_joint_r2.png", dpi=150)
    print("\nwrote artifacts/within_iface_joint_r2.{csv,png}")


if __name__ == "__main__":
    main()
