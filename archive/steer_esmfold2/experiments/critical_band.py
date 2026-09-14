#!/usr/bin/env python3
"""critical_band.py — dissect the band where selection actually matters.

purity_vs_head.py localised the entire Protenix-vs-ESM selector gap to interfaces whose pool is
10-50% acceptable -- mostly wrong, but holding a real minority of correct poses:

    pool purity     Protenix    ESM
    0-10%              23%      26%      both fail (nothing to find)
    10-25%             83%      25%      <-- the gap
    25-50%             73%      45%      <-- the gap
    50-75%            100%     100%      both trivially succeed
    75-100%           100%     100%

and showed ESM's AVERAGE discrimination is as good as Protenix's (AUC 0.797 vs 0.785) despite
argmax accuracy of 67% vs 83%. So ESM's confidence is well-ordered in bulk and wrong at the top.

THE SHARP HYPOTHESIS THIS SCRIPT TESTS:

    ESMFold2's ipTM is POSITIVELY correlated with correctness across the pool,
    but NEGATIVELY correlated WITHIN ITS OWN TOP DECILE.

If true, that is the confident-decoy disease stated exactly, and it explains why every
aggregation scheme we tried today failed: they all read the top of a ranking whose top is
inverted. It also predicts a concrete, testable fix -- do NOT take the most confident pose.

Measures, restricted to the critical band and computed per interface:
  * Cliff's delta (ipTM: acceptable vs not) over the WHOLE pool
  * Cliff's delta over the TOP DECILE only            <- the crux
  * Spearman(confidence, DockQ) whole pool vs top decile
  * DockQ percentile of the argmax-confidence pose    <- where does the winner actually sit?
"""
import collections, csv, glob, os, re
import numpy as np
from scipy.stats import spearmanr

ACC = 0.23
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ESM_CSV = f"{REPO}/results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
PRO_CSV = f"{REPO}/results/csv_files/dockq_ost_protenix_full.csv"
PRO_LOGS = "/home/ebrahim/Protenix/foldbench_abag/predictions_template/gpu*.log"
SCORE = "pair_iptm"
BAND = (0.10, 0.50)          # pool purity band where selection is neither trivial nor hopeless


def template_status():
    pend, seen = [], collections.defaultdict(set)
    for path in sorted(glob.glob(PRO_LOGS)):
        for line in open(path, errors="ignore"):
            m = re.search(r"Found (\d+) templates", line)
            if m:
                pend.append(int(m.group(1))); continue
            m = re.search(r"\]\s+([0-9a-z]{4})\s+\[seed:(\d+)\]", line)
            if m:
                if pend: seen[m.group(1)].update(pend)
                pend = []
    return {c: (max(v) > 0) for c, v in seen.items()}


def load(path):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get("dockq_status") not in (None, "", "ok"):
            continue
        try:
            dq, sc = float(r["dockq"]), float(r[SCORE])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isnan(dq) or np.isnan(sc):
            continue
        by[(r["assembly_id"].split("-")[0], r["chain1"], r["chain2"])].append((r["seed"], dq, sc))
    return by


def cliffs(sc, good):
    """Cliff's delta of confidence between acceptable and non-acceptable poses.
    +1 = every good pose outscores every bad one.  -1 = perfectly backwards.  0 = no signal."""
    g, b = sc[good], sc[~good]
    if len(g) == 0 or len(b) == 0:
        return None
    gt = (g[:, None] > b[None, :]).sum()
    lt = (g[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(g) * len(b)))


def analyse(poses, decile=0.10):
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    good = dq > ACC
    base = good.mean()
    if not (BAND[0] <= base < BAND[1]):
        return None
    order = np.argsort(-sc)
    k = max(3, int(round(decile * len(dq))))
    top = order[:k]

    out = dict(base=base, n=len(dq))
    out["cliff_all"] = cliffs(sc, good)
    out["cliff_top"] = cliffs(sc[top], good[top])          # <- the crux
    out["rho_all"] = spearmanr(sc, dq).statistic
    out["rho_top"] = spearmanr(sc[top], dq[top]).statistic if len(set(sc[top])) > 2 else np.nan
    # where does the winner sit in the pool's DockQ distribution? 1.0 = the single best pose.
    win = int(np.argmax(sc))
    out["win_pctile"] = float((dq < dq[win]).mean())
    out["win_hit"] = bool(dq[win] > ACC)
    out["top_purity"] = float(good[top].mean())
    return out


def main():
    tmpl = template_status()
    esm, pro = load(ESM_CSV), load(PRO_CSV)
    clean = [k for k in sorted(set(esm) & set(pro)) if not tmpl.get(k[0], False)]
    rng = np.random.default_rng(20260713)

    rows = {"Protenix": [], "ESMFold2": []}
    for k in clean:
        seeds = sorted({p[0] for p in esm[k]})
        pick = set(rng.choice(seeds, min(50, len(seeds)), replace=False))   # depth-match
        e = analyse([p for p in esm[k] if p[0] in pick])
        p = analyse(pro[k])
        if e: rows["ESMFold2"].append((k, e))
        if p: rows["Protenix"].append((k, p))

    print(f"SELECTION-CRITICAL BAND: pools that are {BAND[0]:.0%}-{BAND[1]:.0%} acceptable")
    print(f"  (depth-matched 250 poses; raw pair_iptm; zero-template interfaces)\n")
    print(f"{'model':<11}{'n ifaces':>9}{'selector':>10}{'Cliff d':>10}{'Cliff d':>10}"
          f"{'rho':>8}{'rho':>8}{'winner DockQ':>14}")
    print(f"{'':<11}{'':>9}{'(argmax)':>10}{'ALL poses':>10}{'TOP 10%':>10}"
          f"{'ALL':>8}{'TOP':>8}{'percentile':>14}")
    print("-" * 82)
    for m, rs in rows.items():
        if not rs:
            continue
        v = [r[1] for r in rs]
        ca = np.array([x["cliff_all"] for x in v], float)
        ct = np.array([x["cliff_top"] for x in v], float)
        ra = np.array([x["rho_all"] for x in v], float)
        rt = np.array([x["rho_top"] for x in v], float)
        hit = np.array([x["win_hit"] for x in v])
        wp = np.array([x["win_pctile"] for x in v])
        print(f"{m:<11}{len(v):>9}{hit.mean():>10.0%}{np.nanmean(ca):>+10.3f}{np.nanmean(ct):>+10.3f}"
              f"{np.nanmean(ra):>+8.3f}{np.nanmean(rt):>+8.3f}{np.nanmean(wp):>13.0%}")

    print("\n" + "=" * 82)
    print("THE TEST: is confidence positively signed over the pool but NEGATIVE at its own top?")
    print("=" * 82)
    for m, rs in rows.items():
        if not rs:
            continue
        v = [r[1] for r in rs]
        # cliff_top is None when the top decile happens to be all-good or all-bad (no pair to
        # compare). Those interfaces carry no information about top-end ordering -- drop them.
        ca = np.nanmean([x["cliff_all"] for x in v if x["cliff_all"] is not None])
        cts = [x["cliff_top"] for x in v if x["cliff_top"] is not None]
        ct = np.nanmean(cts)
        neg = np.mean([c < 0 for c in cts])
        print(f"\n  {m}")
        print(f"    Cliff's delta over ALL poses   : {ca:+.3f}   "
              f"({'points the RIGHT way' if ca > 0 else 'points the WRONG way'})")
        print(f"    Cliff's delta in its TOP 10%   : {ct:+.3f}   "
              f"({'still RIGHT' if ct > 0 else 'INVERTED — more confident = more wrong'})")
        print(f"    interfaces whose top decile is INVERTED (delta < 0): {neg:.0%}  "
              f"(n={len(cts)} with a mixed top decile)")
        if ca > 0.2 and ct < 0.1:
            print(f"    >>> CONFIRMED: confidence discriminates in bulk and DIES at the top.")

    print("\n\nPER-INTERFACE DETAIL (ESMFold2, critical band, sorted by top-decile Cliff's delta)")
    print(f"{'interface':<12}{'purity':>8}{'Cliff ALL':>11}{'Cliff TOP':>11}{'top-10% purity':>16}{'won?':>6}")
    print("-" * 66)
    for k, x in sorted(rows["ESMFold2"], key=lambda r: (r[1]["cliff_top"] if r[1]["cliff_top"] is not None else 99)):
        name = f"{k[0]}_{k[1]}{k[2]}"
        ct = x["cliff_top"]
        print(f"{name:<12}{x['base']:>8.0%}{x['cliff_all']:>+11.3f}"
              f"{(f'{ct:+.3f}' if ct is not None else 'n/a'):>11}"
              f"{x['top_purity']:>15.0%}{('YES' if x['win_hit'] else 'no'):>6}")


if __name__ == "__main__":
    main()
