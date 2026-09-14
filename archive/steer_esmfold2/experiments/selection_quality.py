#!/usr/bin/env python3
"""selection_quality.py — PREVALENCE-FREE measures of selection quality. No generation mixed in.

Ebrahim's objection, and it is the right one:

    "Is your comparison selection quality specifically (per-target ipTM-vs-DockQ ranking /
     oracle-gap) or overall top-1 accuracy (which conflates generation + selection)? If it's top-1,
     part of the gap could still be generation on FoldBench specifically."

Fair. The `selector` statistic used elsewhere -- P(argmax ipTM hits | a good pose exists) -- is
CONDITIONED on coverage, but it is still PREVALENCE-SENSITIVE: with a purer pool, argmax lands on a
good pose more often even with an IDENTICAL ranker. Protenix's pools ARE purer (50% vs 41%), so its
selector number is flattered by its generator.

This script uses only statistics that are INVARIANT to how many good poses a pool happens to
contain. Every one of them asks: given this pool, how well does the model's own confidence ORDER it?

  SPEARMAN rho(ipTM, DockQ)   per target. Pure rank correlation. Prevalence-free by construction.
  AUC                          P(a random good pose outranks a random bad one). 0.5 = no signal.
                               Prevalence-free by construction.
  CLIFF's DELTA                2*AUC - 1. Same content, signed.
  NORMALISED ORACLE GAP        rank of the first ACCEPTABLE pose in the ipTM ranking, divided by
                               pool size. 0.0 = the top-ranked pose is good (perfect). 1.0 = the
                               good pose is dead last. THIS IS THE OBJECTION'S OWN METRIC: it asks
                               how far down you must dig, which does not care how much treasure is
                               buried.
  TOP-K RECALL                 does a good pose appear in the top 1 / 5 / 10% of the ipTM ranking?
                               Reported against the pool's BASE RATE, so a model whose pool is 50%
                               good gets no credit for a top-10% that is 50% good.

If Protenix leads on ALL of these, its selector really is better and the gap is not a generation
artefact. If the gap collapses once prevalence is removed, Ebrahim is right and our "confidence
head" story is really a "Protenix builds better poses on FoldBench" story.

Depth-matched. Raw pair_iptm for both models (no composite -- see confidence_discrimination.py).
"""
import argparse, collections, csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ACC = 0.23
REPO = Path(__file__).resolve().parents[2]
ARMS = [
    ("ESMFold2-Fast, no MSA", REPO / "results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"),
    ("base ESMFold2, +MSA  ", REPO / "results/csv_files/dockq_ost_esmfold2_msa.csv"),
    ("Protenix,      +MSA  ", REPO / "results/csv_files/dockq_ost_protenix_full.csv"),
]
SCORE = "pair_iptm"


def load(path):
    by = collections.defaultdict(list)
    if not Path(path).exists():
        return by
    for r in csv.DictReader(open(path)):
        if r.get("dockq_status") not in (None, "", "ok"):
            continue
        try:
            dq, sc = float(r["dockq"]), float(r[SCORE])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isnan(dq) or np.isnan(sc):
            continue
        by[(r["assembly_id"][:4], r["chain1"], r["chain2"])].append((r["seed"], dq, sc))
    return by


def quality(poses):
    """Every statistic here is invariant to the POOL'S BASE RATE."""
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    good = dq > ACC
    if not good.any() or good.all():
        return None                              # ranking quality undefined with only one class
    order = np.argsort(-sc)                      # most confident first
    ranks = np.where(good[order])[0]             # positions of the good poses in the ipTM ranking
    n = len(dq)
    g, b = sc[good], sc[~good]
    auc = float(((g[:, None] > b[None, :]).sum() + 0.5 * (g[:, None] == b[None, :]).sum())
                / (len(g) * len(b)))
    out = dict(
        n=n, base=float(good.mean()),
        rho=float(spearmanr(sc, dq).statistic),           # pure rank correlation
        auc=auc,
        cliff=2 * auc - 1,
        oracle_gap=float(ranks[0] / n),                   # how far down the FIRST good pose sits
        oracle_rank=int(ranks[0] + 1),
    )
    # top-k recall vs the base rate: a model whose pool is 50% good earns no credit for a
    # top-10% that is merely 50% good. LIFT = precision@k / base rate.
    for k in (0.01, 0.05, 0.10):
        m = max(1, int(round(k * n)))
        prec = float(good[order[:m]].mean())
        out[f"lift@{k:.0%}"] = prec / out["base"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=20)
    ap.add_argument("--full", action="store_true",
                    help="ESMFold2-Fast vs Protenix on the FULL common benchmark (148 interfaces) "
                         "instead of the 15-interface set that also has the +MSA arm")
    a = ap.parse_args()
    rng = np.random.default_rng(20260713)
    arms = [ARMS[0], ARMS[2]] if a.full else ARMS
    pools = {n: load(p) for n, p in arms}
    keys = sorted(set.intersection(*[set(pools[n]) for n, _ in arms]))
    if not a.full:
        CX = "7txw 7zoz 8che 8fxv 8ido 8oni 8opr 8q6r 8qz3".split()
        keys = [k for k in keys if k[0] in CX]

    rows = {n: [] for n, _ in arms}
    for k in keys:
        per = {}
        for n, _ in arms:
            seeds = sorted({p[0] for p in pools[n][k]})
            if "base ESMFold2" in n:
                keep = {str(s) for s in range(a.nseeds)}
            else:
                keep = set(rng.choice(seeds, min(a.nseeds, len(seeds)), replace=False))
            per[n] = quality([p for p in pools[n][k] if p[0] in keep])
        if all(per[n] for n, _ in arms):          # only targets scoreable for EVERY arm
            for n, _ in arms:
                rows[n].append(per[n])

    n_t = len(rows[arms[0][0]])
    print(f"{n_t} targets scoreable for every arm "
          f"(both classes present; depth-matched to {a.nseeds} seeds)\n")
    print("ALL OF THESE ARE PREVALENCE-FREE — they measure ORDERING, not how much is there to find.\n")
    print(f"{'arm':<24}{'pool purity':>12}{'Spearman':>10}{'AUC':>8}{'Cliff d':>9}"
          f"{'ORACLE GAP':>12}{'lift@1%':>9}{'lift@10%':>10}")
    print(f"{'':<24}{'(generation)':>12}{'rho':>10}{'':>8}{'':>9}{'(lower=better)':>12}{'':>9}{'':>10}")
    print("-" * 94)
    for n, _ in arms:
        v = rows[n]
        print(f"{n:<24}{np.mean([x['base'] for x in v]):>12.0%}"
              f"{np.mean([x['rho'] for x in v]):>+10.3f}"
              f"{np.mean([x['auc'] for x in v]):>8.3f}"
              f"{np.mean([x['cliff'] for x in v]):>+9.3f}"
              f"{np.mean([x['oracle_gap'] for x in v]):>12.1%}"
              f"{np.mean([x['lift@1%'] for x in v]):>9.2f}"
              f"{np.mean([x['lift@10%'] for x in v]):>10.2f}")

    print(f"\n  median ORACLE RANK (position of the first acceptable pose in the ipTM ranking):")
    for n, _ in arms:
        v = rows[n]
        print(f"    {n:<24} rank {int(np.median([x['oracle_rank'] for x in v])):>3} "
              f"of {int(np.median([x['n'] for x in v]))} poses")

    # paired bootstrap on every prevalence-free statistic, ESM(last-but-one) vs Protenix(last)
    a_name, b_name = arms[0][0], arms[-1][0]
    print(f"\n{'=' * 94}")
    print(f"PAIRED BOOTSTRAP over targets:  {b_name.strip()}  −  {a_name.strip()}")
    print("=" * 94)
    print(f"{'statistic':<16}{'difference':>12}{'95% CI':>22}{'P(Protenix better)':>21}")
    print("-" * 71)
    for stat, better_is_high in (("rho", True), ("auc", True), ("cliff", True),
                                 ("oracle_gap", False), ("lift@1%", True), ("lift@10%", True)):
        x = np.array([r[stat] for r in rows[a_name]], float)
        y = np.array([r[stat] for r in rows[b_name]], float)
        d = np.nanmean(y) - np.nanmean(x)
        bs = np.array([(lambda i: np.nanmean(y[i]) - np.nanmean(x[i]))(
            rng.integers(0, len(x), len(x))) for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = (bs < 0).mean() if not better_is_high else (bs > 0).mean()
        flag = "  ***" if p > 0.975 else ("  ." if p > 0.90 else "")
        print(f"{stat:<16}{d:>+12.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{p:>21.3f}{flag}")

    print("""
  ORACLE GAP is the objection's own metric: how far down the ipTM ranking you must dig before you
  hit an acceptable pose, as a FRACTION of the pool. It cannot be inflated by having more good poses
  -- a purer pool makes the denominator and the numerator move together. If Protenix's oracle gap is
  smaller, its confidence genuinely ORDERS better, and the advantage is not a generation artefact.

  LIFT@k = precision in the top k% divided by the pool's base rate. A model whose pool is 50% good
  earns NO credit for a top-10% that is merely 50% good. Lift 1.0 = ranking is worthless.""")


if __name__ == "__main__":
    main()
