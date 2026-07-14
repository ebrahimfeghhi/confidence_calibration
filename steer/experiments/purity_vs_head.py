#!/usr/bin/env python3
"""purity_vs_head.py — is Protenix's selector advantage a BETTER CONFIDENCE HEAD, or just a
PURER POOL?

The dissociation from confidence_top_end.py:

                          selector (argmax)   AUC (discrimination)   base rate (pool purity)
    Protenix  (250)             83%                 0.798                    58%
    ESMFold2  (250, matched)    61%                   --                     48%
    ESMFold2  (5125, full)      56%                 0.772                    43%

Argmax precision is driven by BOTH discrimination and prevalence. Two models with the SAME AUC
will still differ at the top if one's pool contains more correct poses to begin with -- there is
simply more truth to find. Protenix's AUC is only 0.026 higher than ESM's, but its pools are
15 points purer. So the 27-point selector gap might be almost entirely PURITY, and have nothing
to do with the confidence head being a better discriminator.

That distinction decides the project's direction:
  * If it's the HEAD  -> build/borrow a discriminator. Confidence can be fixed.
  * If it's PURITY    -> the generator is the problem. No selector can fix a pool that is mostly
                         wrong, and our whole label-free-selection programme is aimed at the
                         wrong target.

THREE TESTS

  T1  STRATIFY. Bin interfaces by base rate. Compare selector accuracy WITHIN each bin. If the
      gap collapses inside strata, purity explains it.

  T2  MATCHED PAIRS. For each Protenix interface, find ESM interfaces with a base rate within
      +/-5 points, and compare. A non-parametric version of T1 that doesn't depend on bin edges.

  T3  COUNTERFACTUAL. Hold each model's own AUC fixed and ask what its selector accuracy WOULD be
      at the other model's base rate, by resampling its pool to that purity. This isolates the
      head from the pool directly.

All at DEPTH-MATCHED 50 seeds, raw pair_iptm for both models, zero-template interfaces only.
"""
import collections, csv, glob, os, re
import numpy as np

ACC = 0.23
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ESM_CSV = f"{REPO}/results/csv_files/dockq_ost_esmfold2fast_tts_deep.csv"
PRO_CSV = f"{REPO}/results/csv_files/dockq_ost_protenix_full.csv"
PRO_LOGS = "/home/ebrahim/Protenix/foldbench_abag/predictions_template/gpu*.log"
SCORE = "pair_iptm"


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


def stats(poses):
    dq = np.array([p[1] for p in poses]); sc = np.array([p[2] for p in poses])
    good = dq > ACC
    if not good.any():
        return None                                    # NO-POSE: selector undefined
    g, b = sc[good], sc[~good]
    a = None
    if len(b):
        a = float(((g[:, None] > b[None, :]).sum() + 0.5 * (g[:, None] == b[None, :]).sum())
                  / (len(g) * len(b)))
    return dict(base=good.mean(), hit=bool(dq[int(np.argmax(sc))] > ACC), auc=a,
                dq=dq, sc=sc, n=len(dq))


def main():
    rng = np.random.default_rng(20260713)
    tmpl = template_status()
    esm, pro = load(ESM_CSV), load(PRO_CSV)
    clean = [k for k in sorted(set(esm) & set(pro)) if not tmpl.get(k[0], False)]

    # depth-match ESM to Protenix: 50 seeds
    P, E = {}, {}
    for k in clean:
        seeds = sorted({p[0] for p in esm[k]})
        pick = set(rng.choice(seeds, min(50, len(seeds)), replace=False))
        se = stats([p for p in esm[k] if p[0] in pick])
        sp = stats(pro[k])
        if se and sp:
            E[k], P[k] = se, sp
    keys = sorted(E)
    print(f"{len(keys)} zero-template interfaces where BOTH models have >=1 acceptable pose")
    print(f"  (depth-matched: 50 seeds x 5 samples = 250 poses each; raw pair_iptm both)\n")

    pb = np.array([P[k]["base"] for k in keys]); ph = np.array([P[k]["hit"] for k in keys])
    eb = np.array([E[k]["base"] for k in keys]); eh = np.array([E[k]["hit"] for k in keys])
    pa = np.array([P[k]["auc"] for k in keys], float); ea = np.array([E[k]["auc"] for k in keys], float)
    print(f"{'':<12}{'selector':>10}{'base rate':>12}{'AUC':>9}")
    print(f"{'Protenix':<12}{ph.mean():>10.0%}{pb.mean():>12.0%}{np.nanmean(pa):>9.3f}")
    print(f"{'ESMFold2':<12}{eh.mean():>10.0%}{eb.mean():>12.0%}{np.nanmean(ea):>9.3f}")

    # ---------------- T1: stratify by base rate ----------------
    print("\n" + "=" * 78)
    print("T1  SELECTOR ACCURACY WITHIN BASE-RATE STRATA")
    print("    if the gap vanishes inside a stratum, POOL PURITY explains it, not the head")
    print("=" * 78)
    edges = [0.0, 0.10, 0.25, 0.50, 0.75, 1.01]
    print(f"{'base rate':<16}{'Protenix sel':>14}{'(n)':>6}{'ESM sel':>10}{'(n)':>6}{'gap':>8}")
    print("-" * 62)
    for lo, hi in zip(edges[:-1], edges[1:]):
        pm = (pb >= lo) & (pb < hi)
        em = (eb >= lo) & (eb < hi)
        if pm.sum() < 3 or em.sum() < 3:
            continue
        gap = ph[pm].mean() - eh[em].mean()
        print(f"{f'{lo:.0%}-{hi:.0%}':<16}{ph[pm].mean():>13.0%}{pm.sum():>7}"
              f"{eh[em].mean():>10.0%}{em.sum():>6}{gap:>+8.0%}")

    # ---------------- T2: matched pairs ----------------
    print("\n" + "=" * 78)
    print("T2  MATCHED PAIRS  (each Protenix interface vs ESM interfaces of the same purity)")
    print("=" * 78)
    for tol in (0.05, 0.10):
        dp, de = [], []
        for i, k in enumerate(keys):
            m = np.abs(eb - pb[i]) <= tol
            if m.sum() == 0:
                continue
            dp.append(ph[i]); de.append(eh[m].mean())
        print(f"  purity matched to +/-{tol:.0%}: Protenix {np.mean(dp):.0%}  vs  ESM {np.mean(de):.0%}"
              f"   gap {np.mean(dp)-np.mean(de):+.0%}   (n={len(dp)} Protenix interfaces)")

    # ---------------- T3: counterfactual purity ----------------
    print("\n" + "=" * 78)
    print("T3  COUNTERFACTUAL — hold each model's HEAD fixed, swap the POOL PURITY")
    print("    resample each pool to the OTHER model's base rate, keep its own confidence values,")
    print("    and re-run argmax. Isolates the head from the pool.")
    print("=" * 78)

    def resample_to(s, target, reps=200):
        """Keep this model's own (dockq, confidence) pairs, but rebuild the pool so that a
        `target` fraction of poses are acceptable. Confidence values travel WITH their poses,
        so the head's discrimination ability is untouched -- only prevalence changes."""
        dq, sc, n = s["dq"], s["sc"], s["n"]
        gi = np.where(dq > ACC)[0]; bi = np.where(dq <= ACC)[0]
        if len(gi) == 0 or len(bi) == 0:
            return None
        ng = max(1, int(round(target * n))); nb = max(1, n - ng)
        hits = 0
        for _ in range(reps):
            g = rng.choice(gi, ng, replace=True)
            b = rng.choice(bi, nb, replace=True)
            idx = np.concatenate([g, b])
            hits += dq[idx[int(np.argmax(sc[idx]))]] > ACC
        return hits / reps

    e_at_p = [x for x in (resample_to(E[k], pb.mean()) for k in keys) if x is not None]
    p_at_e = [x for x in (resample_to(P[k], eb.mean()) for k in keys) if x is not None]
    print(f"  ESM head, at ESM's own purity ({eb.mean():.0%})        : {eh.mean():.0%}   [observed]")
    print(f"  ESM head, at PROTENIX's purity ({pb.mean():.0%})       : {np.mean(e_at_p):.0%}   "
          f"<- what ESM would score if its pool were as pure")
    print(f"  Protenix head, at its own purity ({pb.mean():.0%})     : {ph.mean():.0%}   [observed]")
    print(f"  Protenix head, at ESM's purity ({eb.mean():.0%})       : {np.mean(p_at_e):.0%}   "
          f"<- what Protenix would score on a pool as dirty as ESM's")
    print()
    obs = ph.mean() - eh.mean()
    fair = np.mean(p_at_e) - eh.mean()          # both heads at ESM purity
    print(f"  observed gap                         : {obs:+.0%}")
    print(f"  gap with purity EQUALISED (both at ESM's {eb.mean():.0%}) : {fair:+.0%}")
    if abs(obs) > 1e-9:
        print(f"  -> {100*(1-fair/obs):.0f}% of the selector gap is POOL PURITY;"
              f" {100*fair/obs:.0f}% is the CONFIDENCE HEAD")


if __name__ == "__main__":
    main()
