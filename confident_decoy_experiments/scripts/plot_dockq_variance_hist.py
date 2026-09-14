#!/usr/bin/env python
"""Distribution of per-interface pose-to-pose variance: ab-ag vs protein-protein.

One count per INTERFACE. Take that interface's 50 seeds x 5 diffusion samples = 250 poses, compute the
variance of the chosen metric across them, histogram those numbers across interfaces.

EXACT SEED SET, ENFORCED. Every dataset is pinned to a fixed list of 50 seeds and an interface missing
ANY seed, or any of the 5 samples within a seed, is dropped and REPORTED. Variance is precisely the
statistic a ragged pose count distorts, so it is not computed over whatever happens to be present.
Seeds cannot be literally shared across datasets (PXM was generated at 300000-300049, FoldBench at
101-150), but within a dataset every retained interface contributes the identical 250 poses.

WHAT THIS DROPS, and why it is not a generation gap. Three protein-protein interfaces come up short:
7quv A|C, 7xpt A|A-2, 8p9c B|C. All three complexes have their full 250 CIFs on disk and OST scored
every pose -- the shortfall is that OST's REFERENCE chain labels are not stable across poses in
homomeric assemblies. 7xpt's homodimer is reported as A|A-2 in most poses and B-3|B-4 in others (same
interface, DockQ 0.842 vs 0.828), while 7quv reports three interfaces A|B, A|C, B|C in some poses and
only two A|B, B|C-2 in others. 11 of 224 pp complexes (5%) show label variation of this kind.
Re-running seeds would not fix it; a fresh pose enters the same lottery. Dropping the incomplete
interfaces is the conservative choice here, and the count is printed so it is never invisible.

PLOTTED ON THE SD SCALE, not variance. Dashed line = median across interfaces, solid = mean.
SD is in the metric's own units (an SD of 0.11 means a typical pose sits ~0.11 DockQ from that
interface's average), whereas variance is in DockQ^2. Squaring also crushes everything below 1 toward
zero, which turned every histogram into a spike at the origin, and it inflates ratios: the ab-ag vs
protein-protein gap is 15x in variance but 3.9x in SD -- the same fact, stated without the squaring.
VARIANCE IS STILL PRINTED in the summary table, because variance is the additive scale: any later
between-seed vs within-seed decomposition has to be done there, not on SDs.

A var/[m(1-m)] normalisation was tried and DROPPED. The motivation was real -- all these metrics live
on [0,1], so var <= m(1-m), and an interface that always scores ~0 or ~1 is forced to have low
variance, which matters because the three sets sit at different means. But the ratio was worse than
the problem: its denominator vanishes exactly where the model always fails, so interfaces with mean
near 0 divide by almost nothing and dominate the median. That artefact, not any real effect, produced
a spurious 2.4x epitope-vs-paratope gap. It is also not "normalised variance" at all --
var/[m(1-m)] = 1 - E[x(1-x)]/[m(1-m)], i.e. a bimodality index that equals 1 for a 0/1 coin flip.

THE LEVEL CONFOUND IS THEREFORE NOT CORRECTED HERE, only declared: read these histograms alongside the
means in the printout, and to compare sets at equal quality use a mean-matched (stratified) comparison
rather than any ratio.

METRICS (--metric). DockQ = (fnat + f(iRMSD) + f(LRMSD)) / 3, so the components are on [0,1] too once
the RMSDs pass through DockQ's own squashing function f(x) = 1/(1 + (x/d)^2), d = 1.5 A for iRMSD and
8.5 A for LRMSD. Plotting raw RMSDs would be meaningless against the m(1-m) bound. Verified:
recombining the components reproduces the reported DockQ to 0.0e+00 over 37,000 PXM poses.
  --jaccard instead plots epitope and paratope Jaccard, ab-ag ONLY (protein-protein has no antibody,
  so neither quantity is defined). Jaccard is already on [0,1].

Sources, all per-pose and traceable:
  artifacts/dockq_per_pose_foldbench.csv          scripts/dump_dockq_per_pose.py, from the OST jsons
  artifacts/pp/dockq_pp.csv                       scripts/score_pp.py, from OST
  artifacts/pxm2025/dockq_pxm2025.csv             scripts/score_pxm2025.py, from PXMeter
  artifacts/pxm2025/dockq_components_pxm2025.csv  scripts/dump_dockq_components_pxm.py
  artifacts/{epitope,paratope}_overlap.csv        scripts/{epitope,paratope}_overlap.py
  artifacts/pxm2025/{epitope,paratope}_overlap_pxm.csv   scripts/epitope_overlap_pxm.py --side
FoldBench and pp use OST DockQ, PXM uses PXMeter's, so compare SHAPES rather than absolute values.

Usage:
  .../python scripts/plot_dockq_variance_hist.py [--metric dockq|fnat|irmsd|lrmsd]
  .../python scripts/plot_dockq_variance_hist.py --jaccard
"""
import argparse
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
FB_SEEDS = frozenset(range(101, 151))          # FoldBench ab-ag and protein-protein
PX_SEEDS = frozenset(range(300000, 300050))    # PXM-2025
N_SAMPLE = 5

# DockQ's own transform for the two RMSD terms, so every metric here is on [0, 1].
SCALE = {"irmsd": 1.5, "lrmsd": 8.5}

PAIR = lambda r: tuple(sorted((r["chain1"], r["chain2"])))
ABAG = lambda r: tuple(sorted((r["ab_chain"], r["ag_chain"])))


def sources(metric, jaccard):
    """[(label, colour, path, key_fn, seed_set, field, restrict_to)]."""
    if jaccard:
        return [
            ("epitope   ab-ag FoldBench", "#b3452c",
             f"{EXP}/artifacts/epitope_overlap.csv", ABAG, FB_SEEDS, "jaccard", None),
            ("epitope   ab-ag PXM-2025", "#31688e",
             f"{P}/epitope_overlap_pxm.csv", ABAG, PX_SEEDS, "jaccard", None),
            ("paratope  ab-ag FoldBench", "#8c5a2b",
             f"{EXP}/artifacts/paratope_overlap.csv", ABAG, FB_SEEDS, "jaccard", None),
            ("paratope  ab-ag PXM-2025", "#4a6f8a",
             f"{P}/paratope_overlap_pxm.csv", ABAG, PX_SEEDS, "jaccard", None),
        ]
    # PXM's component table covers EVERY interface in each assembly, so it is restricted to the
    # ab-ag interfaces that dockq_pxm2025.csv actually scored.
    keep = {(r["pdb"], PAIR(r)) for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv"))}
    px = (f"{P}/dockq_pxm2025.csv" if metric == "dockq"
          else f"{P}/dockq_components_pxm2025.csv")
    return [
        ("ab-ag  FoldBench (OST)", "#b3452c",
         f"{EXP}/artifacts/dockq_per_pose_foldbench.csv", ABAG, FB_SEEDS, metric, None),
        ("ab-ag  PXM-2025 (PXMeter)", "#31688e", px, PAIR, PX_SEEDS, metric, keep),
        ("protein-protein  FoldBench (OST)", "#3f7d4e",
         f"{EXP}/artifacts/pp/dockq_pp.csv", PAIR, FB_SEEDS, metric, None),
    ]


def load(path, key, seed_set, field, restrict):
    """-> ({interface: values}, [(interface, missing_seeds, short_seeds)]) on the EXACT seed set."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        if int(r["seed"]) not in seed_set:
            continue
        k = (r["pdb"], key(r))
        if restrict is not None and k not in restrict:
            continue
        v = r.get(field)
        if v in (None, ""):
            continue
        x = float(v)
        d = SCALE.get(field)
        g[k][int(r["seed"])].append(x if d is None else 1.0 / (1.0 + (x / d) ** 2))

    out, dropped = {}, []
    for k, by_seed in g.items():
        missing = len(seed_set) - len(by_seed)
        short = sum(1 for v in by_seed.values() if len(v) != N_SAMPLE)
        if missing or short:
            dropped.append((k, missing, short))
            continue
        out[k] = np.array([x for v in by_seed.values() for x in v])
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="dockq", choices=("dockq", "fnat", "irmsd", "lrmsd"))
    ap.add_argument("--jaccard", action="store_true")
    A = ap.parse_args()

    lab = ("Jaccard" if A.jaccard else
           {"irmsd": "f(iRMSD)", "lrmsd": "f(LRMSD)", "dockq": "DockQ"}
           .get(A.metric, A.metric))
    data = []
    print(f"exact seed set enforced: 50 seeds x {N_SAMPLE} samples = 250 poses per interface\n")
    for name, col, path, key, seeds, field, restrict in sources(A.metric, A.jaccard):
        vals, dropped = load(path, key, seeds, field, restrict)
        m = np.array([v.mean() for v in vals.values()])
        var = np.array([v.var() for v in vals.values()])
        data.append((name, col, m, var, var / np.maximum(m * (1 - m), 1e-9)))
        note = ""
        if dropped:
            ex = "; ".join(f"{p}_{'|'.join(c)} (-{ms} seeds, {sh} short)"
                           for (p, c), ms, sh in dropped[:3])
            note = f"   DROPPED {len(dropped)} incomplete: {ex}"
        print(f"  {name:<34} kept {len(vals):>4}{note}")

    ncol = len(data)
    fig, axes = plt.subplots(1, ncol, figsize=(5.5 * ncol, 5.0), squeeze=False)
    hi = 0.55 if A.jaccard else 0.50
    bins = np.linspace(0, hi, 41)
    for j, (name, col, m, var, _) in enumerate(data):
        ax = axes[0][j]
        a = np.clip(np.sqrt(var), 0, hi)
        ax.hist(a, bins=bins, color=col, alpha=0.55)
        ax.hist(a, bins=bins, histtype="step", lw=1.8, color=col)
        ax.axvline(np.median(a), color="0.20", lw=1.5, ls="--", label=f"median {np.median(a):.3f}")
        ax.axvline(a.mean(), color="0.20", lw=1.5, ls="-", label=f"mean {a.mean():.3f}")
        ax.set_title(f"{name}\nn={len(a)},  mean {lab} = {m.mean():.3f}", fontsize=12)
        ax.set_xlabel(f"SD of {lab} across the interface's 250 poses", fontsize=12)
        if j == 0:
            ax.set_ylabel("number of interfaces", fontsize=12)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, max(np.histogram(a, bins=bins)[0]) * 1.18)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=11, loc="upper right")
    fig.tight_layout()
    out = f"{EXP}/artifacts/{'jaccard' if A.jaccard else A.metric}_sd_hist.png"
    fig.savefig(out, dpi=150)

    print(f"\n{'set':<34}{'n':>5}{'mean ' + lab:>14}{'median SD':>11}{'mean SD':>10}"
          f"{'max SD':>9}{'median var':>12}{'mean var':>11}")
    for name, col, m, var, _ in data:
        sd = np.sqrt(var)
        print(f"{name:<34}{len(m):>5}{m.mean():>14.3f}{np.median(sd):>11.3f}{sd.mean():>10.3f}"
              f"{sd.max():>9.3f}{np.median(var):>12.4f}{var.mean():>11.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
