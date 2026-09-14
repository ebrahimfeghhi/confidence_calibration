"""Does a pose that finds the right epitope also have a better-built antibody?

LEFT   every pose pooled (one dot per pose): epitope Jaccard against paratope RMSD (local).
       Pooled is the MISLEADING view -- easy targets sit low-right (good epitope, low RMSD) and hard
       ones high-left, so between-target differences manufacture a correlation that says nothing
       about re-rolling seeds on a single target. Shown because it is the number people quote.
RIGHT  the honest view: correlate the two WITHIN each interface, then histogram those 148 values.

SIGN: lower RMSD is better, higher Jaccard is better, so NEGATIVE r means "poses that found the
epitope also have a better-built antibody"; r near 0 means antibody geometry is independent of where
the pose docked.

Paratope RMSD superposes within the antibody on a fixed GT-defined residue set, so it does not
depend on docking success -- that independence is what makes the question well posed.

Usage: .../python scripts/plot_epi_jac_vs_para_rmsd.py
"""
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
SEED_LO, SEED_HI = 101, 150
COL = "#b3452c"


def load(fn, cols):
    out = {}
    for r in csv.DictReader(open(f"{EXP}/artifacts/{fn}")):
        if not (SEED_LO <= int(r["seed"]) <= SEED_HI):
            continue
        try:
            out[(r["pdb"], r["ab_chain"], r["ag_chain"], r["seed"], r["sample"])] = \
                tuple(float(r[c]) for c in cols)
        except ValueError:
            pass
    return out


def main():
    epi = load("epitope_overlap.csv", ["jaccard"])
    prm = load("paratope_rmsd.csv", ["para_local"])
    per = collections.defaultdict(list)
    for k in set(epi) & set(prm):
        per[f"{k[0]}_{k[1]}{k[2]}"].append((epi[k][0], prm[k][0]))

    A = np.vstack([np.array(v) for v in per.values()])
    rs = []
    for v in per.values():
        a = np.array(v)
        if len(a) >= 100 and a[:, 0].std() > 1e-9 and a[:, 1].std() > 1e-9:
            rs.append(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    rs = np.array(rs)

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.6))
    ax[0].scatter(A[:, 0], A[:, 1], s=3, c=COL, alpha=0.12, lw=0)
    pooled = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
    m, b = np.polyfit(A[:, 0], A[:, 1], 1)
    gx = np.linspace(0, 1, 50)
    ax[0].plot(gx, m * gx + b, color="0.2", lw=2,
               label=f"pooled r = {pooled:+.3f}   (r$^2$ = {pooled**2:.3f})")
    ax[0].set_xlabel("epitope Jaccard of the pose", fontsize=15)
    ax[0].set_ylabel(u"paratope RMSD, local (Å)", fontsize=15)
    ax[0].set_title(f"every pose pooled   n = {len(A):,}\n"
                    "between-target differences inflate this", fontsize=13)
    ax[0].set_ylim(0, np.percentile(A[:, 1], 99.5))
    ax[0].set_xlim(-0.02, 1.02)

    ax[1].hist(rs, bins=28, color=COL, alpha=0.85)
    ax[1].axvline(0, color="0.35", lw=1.6)
    ax[1].axvline(np.median(rs), color="#1b5e20", lw=2.2, ls="--",
                  label=f"median r = {np.median(rs):+.3f}")
    ax[1].set_xlabel("correlation r within one interface", fontsize=15)
    ax[1].set_ylabel("number of interfaces", fontsize=15)
    ax[1].set_title(f"computed within each interface   n = {len(rs)} interfaces\n"
                    f"{100*(rs<0).mean():.0f}% negative, but median r$^2$ is only "
                    f"{np.median(rs)**2:.3f}", fontsize=13)
    for a in ax:
        a.tick_params(labelsize=13)
        a.grid(alpha=0.2)
        a.legend(loc="upper right", frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{EXP}/artifacts/epi_jac_vs_para_rmsd.png", dpi=150)
    print(f"poses {len(A)}  pooled r {pooled:+.3f}")
    print(f"interfaces {len(rs)}  median r {np.median(rs):+.3f}  "
          f"negative {100*(rs<0).mean():.0f}%")
    print("wrote artifacts/epi_jac_vs_para_rmsd.png")


if __name__ == "__main__":
    main()
